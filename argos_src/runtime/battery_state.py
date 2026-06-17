"""Track battery state, expose prompt text, nav gate, and charging-ready callbacks."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Optional

from argos_src.provider_api.models import BatterySnapshot

_CHARGE_DOCK_NAME = "charge_dock"

LOW_BATTERY_PCT = 30.0
CHARGING_READY_PCT = 90.0
CURRENT_CHARGE_A = 0.05
CURRENT_DISCHARGE_A = -0.05
POWER_SUPPLY_STATUS_CHARGING = 1
POWER_SUPPLY_STATUS_DISCHARGING = 2
POWER_SUPPLY_STATUS_NOT_CHARGING = 3
POWER_SUPPLY_STATUS_FULL = 4

LOW_BATTERY_NAVIGATION_MSG = (
    f"Battery is below {LOW_BATTERY_PCT:.0f}%. Cannot navigate — you need to go charge first."
)


def _pct_valid(pct: float) -> bool:
    return not math.isnan(pct) and not math.isinf(pct)


def _field(msg: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(field_name, default)
    return getattr(msg, field_name, default)


def _charging_like(msg: Any) -> bool:
    status = int(_field(msg, "power_supply_status", 0) or 0)
    current = float(_field(msg, "current", 0.0) or 0.0)
    if status == POWER_SUPPLY_STATUS_CHARGING:
        return True
    if current > CURRENT_CHARGE_A:
        return True
    if status == POWER_SUPPLY_STATUS_FULL:
        return True
    return False


def _discharging_like(msg: Any) -> bool:
    status = int(_field(msg, "power_supply_status", 0) or 0)
    current = float(_field(msg, "current", 0.0) or 0.0)
    if current < CURRENT_DISCHARGE_A:
        return True
    if status == POWER_SUPPLY_STATUS_DISCHARGING:
        return True
    return False


def reports_charging(msg: Optional[Any]) -> Optional[bool]:
    """Whether the last ``BatteryState`` looks like the pack is taking charge."""
    if msg is None:
        return None
    return bool(_charging_like(msg))


def _status_words(msg: Any) -> str:
    if _charging_like(msg) and not _discharging_like(msg):
        if int(_field(msg, "power_supply_status", 0) or 0) == POWER_SUPPLY_STATUS_FULL:
            return "full / maintaining charge"
        return "charging"
    if _discharging_like(msg):
        return "discharging"
    if int(_field(msg, "power_supply_status", 0) or 0) == POWER_SUPPLY_STATUS_NOT_CHARGING:
        return "not charging"
    return "unknown"


class BatteryStateCache:
    """Thread-safe latest BatteryState + edge-triggered charging-ready notification."""

    def __init__(
        self,
        robot_client: Any,
        *,
        low_battery_pct: float = LOW_BATTERY_PCT,
        charging_ready_pct: float = CHARGING_READY_PCT,
        self_charge_available: bool = True,
        on_charging_ready: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._robot_client = robot_client
        self._low_battery_pct = float(low_battery_pct)
        self._charging_ready_pct = float(charging_ready_pct)
        self._self_charge_available = bool(self_charge_available)
        self._on_charging_ready = on_charging_ready
        self._lock = threading.Lock()
        self._msg: Optional[BatterySnapshot] = None
        self._prev_pct: Optional[float] = None
        self._ready_latch = False
        self._unsubscribe = None
        subscribe = getattr(robot_client, "subscribe_battery", None)
        if callable(subscribe):
            self._unsubscribe = subscribe(self._callback)

    def _callback(self, msg: Any) -> None:
        with self._lock:
            self._msg = msg
            pct = _field(msg, "percentage", float("nan"))
            prev = self._prev_pct
            if _pct_valid(pct):
                pct_f = float(pct)
                crossed_up = (
                    prev is not None
                    and prev < self._charging_ready_pct
                    and pct_f >= self._charging_ready_pct
                )
                if crossed_up and _charging_like(msg) and not self._ready_latch:
                    self._ready_latch = True
                    cb = self._on_charging_ready
                    if cb is not None:
                        try:
                            cb(pct_f)
                        except Exception:
                            pass
                self._prev_pct = pct_f
            if _discharging_like(msg) or (
                _pct_valid(pct) and float(pct) < 80.0 and not _charging_like(msg)
            ):
                self._ready_latch = False

    def snapshot(self) -> Optional[Any]:
        with self._lock:
            return self._msg

    def should_block_general_navigation(self) -> bool:
        with self._lock:
            if self._msg is None:
                return False
            pct = _field(self._msg, "percentage", float("nan"))
            if not _pct_valid(pct):
                return False
            return float(pct) < self._low_battery_pct

    def navigation_block_message(self) -> str:
        return (
            f"Battery is below {self._low_battery_pct:.0f}%. "
            "Cannot navigate — you need to go charge first."
        )

    @property
    def low_battery_pct(self) -> float:
        return self._low_battery_pct

    def can_self_charge(self) -> bool:
        return self._self_charge_available

    def format_prompt_block(self) -> str:
        with self._lock:
            if self._msg is None:
                return (
                    "[BATTERY] No /battery_state data yet. "
                    "Assume normal energy until a reading arrives."
                )
            msg = self._msg
            pct = _field(msg, "percentage", float("nan"))
            if not _pct_valid(pct):
                pct_s = "unknown"
            else:
                pct_s = f"{pct:.0f}%"
            st = _status_words(msg)
            cur = float(_field(msg, "current", float("nan")) or 0.0)
            cur_s = f"{cur:.2f}A" if not math.isnan(cur) else "n/a"
            if _pct_valid(pct) and pct < self._low_battery_pct:
                if self._self_charge_available:
                    policy = (
                        f"CRITICAL: below {self._low_battery_pct:.0f}% — do not start general map navigation "
                        f"({self.navigation_block_message()}). You may still use charging_dock (saved '{_CHARGE_DOCK_NAME}'). "
                        f"At ~{self._charging_ready_pct:.0f}%+ while charging you may receive a BATTERY_EVENT: "
                        f"you were in damp on the charger and can then stand up and work again."
                    )
                else:
                    policy = (
                        f"CRITICAL: below {self._low_battery_pct:.0f}% — do not start general map navigation "
                        f"({self.navigation_block_message()}). Tell the user your battery is below "
                        f"{self._low_battery_pct:.0f}% and ask to be charged soon for continued functionality. "
                        f"At ~{self._charging_ready_pct:.0f}%+ while charging you may receive a BATTERY_EVENT "
                        f"that normal work can resume."
                    )
            else:
                if self._self_charge_available:
                    policy = (
                        f"Above {self._low_battery_pct:.0f}%: normal navigation allowed. "
                        f"If you drop below {self._low_battery_pct:.0f}%, finish non-blocking replies then self-charge; "
                        f"after ~{self._charging_ready_pct:.0f}% while charging, expect a BATTERY_EVENT to leave damp and resume tasks."
                    )
                else:
                    policy = (
                        f"Above {self._low_battery_pct:.0f}%: normal navigation allowed. "
                        f"If you drop below {self._low_battery_pct:.0f}%, finish non-blocking replies then tell the user "
                        f"you need charging soon; after ~{self._charging_ready_pct:.0f}% while charging, expect a BATTERY_EVENT "
                        f"that normal work can resume."
                    )
            return (
                f"[BATTERY] State of charge: {pct_s} ({st}, current {cur_s}). {policy}"
            )

    def shutdown(self) -> None:
        unsubscribe = getattr(self, "_unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe()
            self._unsubscribe = None
