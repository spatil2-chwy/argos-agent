"""Background owner-facing turn controller for Argos interactions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

from argos_src.provider_api.errors import is_provider_error
from argos_src.runtime.motion_locks import motion_lock_for_channel

logger = logging.getLogger(__name__)
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_FRAME = "odom"
ROBOT_FRAME = "base_link"


@dataclass(frozen=True)
class OwnerTurnSettings:
    """Runtime knobs for automatic face-bearing turns."""

    enabled: bool = False
    deadband_deg: float = 3.0
    turn_gain: float = 1.0
    max_turn_deg: float = 25.0
    angular_speed_rad_s: float = 0.8
    command_hz: float = 50.0
    delay_after_recording_sec: float = 0.05
    odom_frame: str = ODOM_FRAME
    robot_frame: str = ROBOT_FRAME
    yaw_tolerance_deg: float = 1.5
    max_duration_sec: float = 1.5
    slow_zone_deg: float = 8.0
    min_angular_speed_rad_s: float = 0.25

    def __post_init__(self) -> None:
        if self.deadband_deg < 0.0:
            raise ValueError("deadband_deg must be >= 0")
        if self.turn_gain <= 0.0:
            raise ValueError("turn_gain must be > 0")
        if self.max_turn_deg <= 0.0:
            raise ValueError("max_turn_deg must be > 0")
        if self.angular_speed_rad_s <= 0.0:
            raise ValueError("angular_speed_rad_s must be > 0")
        if self.command_hz <= 0.0:
            raise ValueError("command_hz must be > 0")
        if self.delay_after_recording_sec < 0.0:
            raise ValueError("delay_after_recording_sec must be >= 0")
        if self.yaw_tolerance_deg < 0.0:
            raise ValueError("yaw_tolerance_deg must be >= 0")
        if self.max_duration_sec <= 0.0:
            raise ValueError("max_duration_sec must be > 0")
        if self.slow_zone_deg <= 0.0:
            raise ValueError("slow_zone_deg must be > 0")
        if self.min_angular_speed_rad_s <= 0.0:
            raise ValueError("min_angular_speed_rad_s must be > 0")
        if self.min_angular_speed_rad_s > self.angular_speed_rad_s:
            raise ValueError("min_angular_speed_rad_s must be <= angular_speed_rad_s")


@dataclass(frozen=True)
class OwnerTurnRequest:
    person_id: str
    req_id: str = ""
    owner_source: str = ""


class OwnerTurnController:
    """Turns the robot toward the resolved turn owner without involving the LLM."""

    def __init__(
        self,
        *,
        connector: Any,
        face_service: Any,
        nav_state: Any | None = None,
        recording_state_provider: Callable[[], bool] | None = None,
        settings: OwnerTurnSettings | None = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.connector = connector
        self.face_service = face_service
        self.nav_state = nav_state
        self.recording_state_provider = recording_state_provider
        self.settings = settings or OwnerTurnSettings()
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._stop = threading.Event()
        self._queue: queue.Queue[OwnerTurnRequest] = queue.Queue(maxsize=1)
        self._cancel_lock = threading.Lock()
        self._canceled_req_ids: set[str] = set()
        self._motion_lock = None
        self._motion_lock_acquired = False
        self._last_transform_failure_provider = False
        self._next_warning_log_by_key: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_turn(
        self,
        *,
        person_id: str | None,
        req_id: str = "",
        owner_source: str = "",
    ) -> bool:
        """Queue one owner-facing turn request."""
        if not self.settings.enabled:
            return False
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            return False

        request = OwnerTurnRequest(
            person_id=rendered_person_id,
            req_id=str(req_id or "").strip(),
            owner_source=str(owner_source or "").strip(),
        )
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                return False
        return True

    def cancel_request(self, *, req_id: str, reason: str = "") -> None:
        """Cancel a queued or active owner turn for a specific agent turn."""
        rendered_req_id = str(req_id or "").strip()
        if not rendered_req_id:
            return
        with self._cancel_lock:
            self._canceled_req_ids.add(rendered_req_id)
        logger.debug(
            "Owner turn cancel requested req_id=%s reason=%s",
            rendered_req_id,
            str(reason or "").strip(),
        )

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._execute_request(request)
            except Exception:
                logger.exception(
                    "Owner turn request failed req_id=%s person_id=%s",
                    request.req_id,
                    request.person_id,
                )
            finally:
                self._clear_request_cancel(request)
                self._queue.task_done()

    def _execute_request(self, request: OwnerTurnRequest) -> None:
        delay = float(self.settings.delay_after_recording_sec)
        if delay > 0.0:
            self._sleep_fn(delay)

        if self._is_request_canceled(request):
            logger.debug(
                "Owner turn skipped because request was canceled req_id=%s person_id=%s",
                request.req_id,
                request.person_id,
            )
            return
        if self._is_recording_active():
            logger.debug(
                "Owner turn skipped because recording restarted req_id=%s person_id=%s",
                request.req_id,
                request.person_id,
            )
            return
        if self._nav_goal_active():
            logger.debug(
                "Owner turn skipped because navigation is active req_id=%s person_id=%s",
                request.req_id,
                request.person_id,
            )
            return

        target = self.face_service.get_face_turn_target(request.person_id)
        if target is None:
            logger.debug(
                "Owner turn skipped because owner face target is unavailable req_id=%s person_id=%s",
                request.req_id,
                request.person_id,
            )
            return

        bearing_rad = float(target.bearing_rad)
        deadband_rad = math.radians(float(self.settings.deadband_deg))
        if abs(bearing_rad) <= deadband_rad:
            logger.debug(
                "Owner turn skipped inside deadband req_id=%s person_id=%s bearing_deg=%.2f deadband_deg=%.2f",
                request.req_id,
                request.person_id,
                math.degrees(bearing_rad),
                self.settings.deadband_deg,
            )
            return

        max_turn_rad = math.radians(float(self.settings.max_turn_deg))
        scaled_rad = bearing_rad * float(self.settings.turn_gain)
        target_rad = max(-max_turn_rad, min(max_turn_rad, scaled_rad))
        self._execute_closed_loop_turn(
            request=request,
            target_rad=target_rad,
            bearing_rad=bearing_rad,
        )

    def _execute_closed_loop_turn(
        self,
        *,
        request: OwnerTurnRequest,
        target_rad: float,
        bearing_rad: float,
    ) -> None:
        start_yaw = self._get_robot_yaw_rad()
        if start_yaw is None:
            if self._last_transform_failure_provider:
                self._warn_throttled(
                    "transform_unavailable",
                    "Owner turn skipped because robot provider transform %s -> %s is unavailable req_id=%s person_id=%s",
                    self.settings.odom_frame,
                    self.settings.robot_frame,
                    request.req_id,
                    request.person_id,
                )
            else:
                logger.error(
                    "Owner turn skipped because transform %s -> %s is unavailable req_id=%s person_id=%s",
                    self.settings.odom_frame,
                    self.settings.robot_frame,
                    request.req_id,
                    request.person_id,
                )
            return

        tolerance_rad = math.radians(float(self.settings.yaw_tolerance_deg))
        deadline = self._time_fn() + float(self.settings.max_duration_sec)
        period = 1.0 / float(self.settings.command_hz)
        last_actual_rad = 0.0
        last_remaining_rad = target_rad
        first_publish = True

        try:
            while not self._stop.is_set() and self._time_fn() < deadline:
                if self._is_request_canceled(request):
                    logger.debug(
                        "Owner turn closed-loop interrupted by tool priority req_id=%s person_id=%s",
                        request.req_id,
                        request.person_id,
                    )
                    return
                if self._is_recording_active():
                    logger.debug(
                        "Owner turn closed-loop interrupted by recording req_id=%s person_id=%s",
                        request.req_id,
                        request.person_id,
                    )
                    return
                if self._nav_goal_active():
                    logger.debug(
                        "Owner turn closed-loop interrupted by navigation req_id=%s person_id=%s",
                        request.req_id,
                        request.person_id,
                    )
                    return

                current_yaw = self._get_robot_yaw_rad()
                if current_yaw is None:
                    if self._last_transform_failure_provider:
                        self._warn_throttled(
                            "transform_became_unavailable",
                            "Owner turn stopped because robot provider transform %s -> %s became unavailable req_id=%s person_id=%s actual_deg=%.2f",
                            self.settings.odom_frame,
                            self.settings.robot_frame,
                            request.req_id,
                            request.person_id,
                            math.degrees(last_actual_rad),
                        )
                    else:
                        logger.error(
                            "Owner turn stopped because transform %s -> %s became unavailable req_id=%s person_id=%s actual_deg=%.2f",
                            self.settings.odom_frame,
                            self.settings.robot_frame,
                            request.req_id,
                            request.person_id,
                            math.degrees(last_actual_rad),
                        )
                    return

                actual_rad = _normalize_angle_rad(current_yaw - start_yaw)
                remaining_rad = _normalize_angle_rad(target_rad - actual_rad)
                last_actual_rad = actual_rad
                last_remaining_rad = remaining_rad
                if abs(remaining_rad) <= tolerance_rad:
                    logger.info(
                        "Owner turn complete mode=closed_loop req_id=%s person_id=%s bearing_deg=%.2f gain=%.2f command_deg=%.2f actual_deg=%.2f error_deg=%.2f",
                        request.req_id,
                        request.person_id,
                        math.degrees(bearing_rad),
                        self.settings.turn_gain,
                        math.degrees(target_rad),
                        math.degrees(actual_rad),
                        math.degrees(remaining_rad),
                    )
                    return

                angular_z = self._angular_speed_for_remaining(remaining_rad)
                if not self._publish_velocity(angular_z):
                    logger.debug(
                        "Owner turn skipped because cmd_vel is busy req_id=%s person_id=%s",
                        request.req_id,
                        request.person_id,
                    )
                    return
                if first_publish:
                    first_publish = False
                    logger.info(
                        "Owner turn dispatch mode=closed_loop req_id=%s person_id=%s bearing_deg=%.2f gain=%.2f command_deg=%.2f angular_z=%.2f tolerance_deg=%.2f",
                        request.req_id,
                        request.person_id,
                        math.degrees(bearing_rad),
                        self.settings.turn_gain,
                        math.degrees(target_rad),
                        angular_z,
                        self.settings.yaw_tolerance_deg,
                    )
                self._sleep_fn(period)
        finally:
            if getattr(self, "_motion_lock_acquired", False):
                try:
                    self._publish_stop()
                except Exception as exc:
                    if is_provider_error(exc):
                        self._warn_throttled(
                            "publish_stop_provider",
                            "Owner turn robot provider stop publish failed req_id=%s person_id=%s: %s",
                            request.req_id,
                            request.person_id,
                            exc,
                        )
                    else:
                        logger.exception(
                            "Owner turn failed to publish stop req_id=%s person_id=%s",
                            request.req_id,
                            request.person_id,
                        )
                finally:
                    self._release_motion_lock_if_held()

        if self._stop.is_set():
            logger.debug(
                "Owner turn stopped by shutdown req_id=%s person_id=%s actual_deg=%.2f",
                request.req_id,
                request.person_id,
                math.degrees(last_actual_rad),
            )
            return

        logger.info(
            "Owner turn stopped mode=closed_loop_timeout req_id=%s person_id=%s bearing_deg=%.2f gain=%.2f command_deg=%.2f actual_deg=%.2f error_deg=%.2f",
            request.req_id,
            request.person_id,
            math.degrees(bearing_rad),
            self.settings.turn_gain,
            math.degrees(target_rad),
            math.degrees(last_actual_rad),
            math.degrees(last_remaining_rad),
        )

    def _angular_speed_for_remaining(self, remaining_rad: float) -> float:
        abs_remaining = abs(float(remaining_rad))
        slow_zone_rad = math.radians(float(self.settings.slow_zone_deg))
        max_speed = float(self.settings.angular_speed_rad_s)
        min_speed = float(self.settings.min_angular_speed_rad_s)
        ratio = min(1.0, max(0.0, abs_remaining / slow_zone_rad))
        speed = min_speed + ((max_speed - min_speed) * ratio)
        return math.copysign(speed, remaining_rad)

    def _get_robot_yaw_rad(self) -> float | None:
        getter = getattr(self.connector, "get_transform", None)
        if not callable(getter):
            self._last_transform_failure_provider = False
            return None
        try:
            transform = getter(self.settings.odom_frame, self.settings.robot_frame)
        except Exception as exc:
            self._last_transform_failure_provider = is_provider_error(exc)
            if self._last_transform_failure_provider:
                self._warn_throttled(
                    "read_transform_provider",
                    "Owner turn robot provider transform read failed %s -> %s: %s",
                    self.settings.odom_frame,
                    self.settings.robot_frame,
                    exc,
                )
            else:
                logger.exception(
                    "Owner turn failed to read transform %s -> %s",
                    self.settings.odom_frame,
                    self.settings.robot_frame,
                )
            return None
        self._last_transform_failure_provider = False
        if hasattr(transform, "rotation") and isinstance(transform.rotation, tuple):
            x, y, z, w = transform.rotation
            return _yaw_from_quaternion(float(x), float(y), float(z), float(w))
        rotation = getattr(getattr(transform, "transform", None), "rotation", None)
        if rotation is None:
            return None
        return _yaw_from_quaternion(
            float(getattr(rotation, "x", 0.0) or 0.0),
            float(getattr(rotation, "y", 0.0) or 0.0),
            float(getattr(rotation, "z", 0.0) or 0.0),
            float(getattr(rotation, "w", 1.0) or 1.0),
        )

    def _publish_velocity(self, angular_z: float) -> bool:
        if not self._acquire_motion_lock():
            return False
        try:
            self._send_velocity_sample(angular_z=float(angular_z))
            return True
        except Exception as exc:
            if is_provider_error(exc):
                self._warn_throttled(
                    "publish_velocity_provider",
                    "Owner turn robot provider velocity publish failed: %s",
                    exc,
                )
                self._release_motion_lock_if_held()
                return False
            raise

    def _publish_stop(self) -> None:
        self._send_velocity_sample(angular_z=0.0)

    def _send_velocity_sample(self, *, angular_z: float) -> None:
        publisher = getattr(self.connector, "publish_velocity", None)
        if not callable(publisher):
            raise AttributeError(
                "OwnerTurnController connector must provide publish_velocity"
            )
        publisher(angular_z=float(angular_z))

    def _acquire_motion_lock(self) -> bool:
        if getattr(self, "_motion_lock_acquired", False):
            return True
        lock = motion_lock_for_channel(CMD_VEL_TOPIC)
        acquired = lock.acquire(blocking=False)
        if acquired:
            self._motion_lock_acquired = True
            self._motion_lock = lock
        return acquired

    def _warn_throttled(self, key: str, message: str, *args: object) -> None:
        now = time.monotonic()
        next_allowed = self._next_warning_log_by_key.get(key, 0.0)
        if now < next_allowed:
            return
        self._next_warning_log_by_key[key] = now + 10.0
        logger.warning(message, *args)

    def _release_motion_lock_if_held(self) -> None:
        if not getattr(self, "_motion_lock_acquired", False):
            return
        lock = getattr(self, "_motion_lock", None)
        self._motion_lock_acquired = False
        if lock is not None:
            lock.release()

    def _is_recording_active(self) -> bool:
        provider = self.recording_state_provider
        if provider is None:
            return False
        try:
            return bool(provider())
        except Exception:
            logger.exception("Owner turn recording-state provider failed")
            return True

    def _nav_goal_active(self) -> bool:
        nav_state = self.nav_state
        if nav_state is None:
            return False
        getter = getattr(nav_state, "get_active_goal", None)
        if not callable(getter):
            return False
        try:
            return getter() is not None
        except Exception:
            logger.exception("Owner turn navigation-state check failed")
            return True

    def _is_request_canceled(self, request: OwnerTurnRequest) -> bool:
        req_id = str(request.req_id or "").strip()
        if not req_id:
            return False
        with self._cancel_lock:
            return req_id in self._canceled_req_ids

    def _clear_request_cancel(self, request: OwnerTurnRequest) -> None:
        req_id = str(request.req_id or "").strip()
        if not req_id:
            return
        with self._cancel_lock:
            self._canceled_req_ids.discard(req_id)


def _normalize_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)
