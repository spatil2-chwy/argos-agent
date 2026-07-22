"""Load/save named locations (map frame x, y, theta) and shared navigation state."""

from dataclasses import dataclass
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

# Saved pose name used by charging_dock and low-battery navigation exceptions.
CHARGE_DOCK_LOCATION_NAME = "charge_dock"

# Type for a single location: x, y, theta in map frame
LocationCoords = dict[str, float]  # {"x": float, "y": float, "theta": float}

NAV_RESULT_DELIVERY_MODEL_EVENT = "model_event"
NAV_RESULT_DELIVERY_TOOL_RESULT = "tool_result"
NAV_RESULT_DELIVERY_RUNTIME_ONLY = "runtime_only"


@dataclass(frozen=True)
class NavigationPolicy:
    """Execution policy for the active navigation goal."""

    source: str
    interruptible: bool = True
    passive_listen_allowed: bool = True
    result_delivery: str = NAV_RESULT_DELIVERY_MODEL_EVENT

    def allows_auto_interrupt(self) -> bool:
        return self.interruptible

    def allows_passive_listen(self) -> bool:
        return self.passive_listen_allowed

    def allows_proactive_face_attention(self) -> bool:
        return self.passive_listen_allowed


INTERRUPTIBLE_NAVIGATION_POLICY = NavigationPolicy(
    source="general_navigation",
    interruptible=True,
    passive_listen_allowed=True,
)
FOCUSED_NAVIGATION_POLICY = NavigationPolicy(
    source="human_task",
    interruptible=False,
    passive_listen_allowed=False,
    result_delivery=NAV_RESULT_DELIVERY_TOOL_RESULT,
)
CHARGING_DOCK_NAVIGATION_POLICY = NavigationPolicy(
    source="charging_dock",
    interruptible=False,
    passive_listen_allowed=False,
    result_delivery=NAV_RESULT_DELIVERY_TOOL_RESULT,
)
PATROL_NAVIGATION_POLICY = NavigationPolicy(
    source="patrol",
    interruptible=True,
    passive_listen_allowed=True,
    result_delivery=NAV_RESULT_DELIVERY_RUNTIME_ONLY,
)


def _locations_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "nav_locations"


def resolve_map_locations_path(map_filename: str) -> Path:
    """Resolve map-specific locations file under resources/nav_locations/.

    The filename must be a basename ending with `.json` (no path separators).
    """
    if not map_filename.endswith(".json"):
        raise ValueError(
            f"Map locations filename must end with .json, got: {map_filename}"
        )

    basename = os.path.basename(map_filename)
    if basename != map_filename:
        raise ValueError(
            "Map locations filename must be a bare filename (no directories)."
        )

    return _locations_dir() / basename


def _default_path() -> Path:
    # If no map is specified, use a neutral file in the map-aware directory.
    return _locations_dir() / "unspecified_map.json"


def ensure_locations_file(path: os.PathLike[str] | str) -> Path:
    """Ensure the locations file exists; create an empty JSON object if missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({}, f, indent=2)
        tmp.replace(p)
    return p


def load_locations(path: Optional[os.PathLike[str] | str] = None) -> dict[str, LocationCoords]:
    """Load locations from JSON. Returns {} if file missing or invalid."""
    p = Path(path) if path is not None else _default_path()
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # Normalize keys and ensure x, y, theta
        out: dict[str, LocationCoords] = {}
        for k, v in data.items():
            if isinstance(v, dict) and "x" in v and "y" in v and "theta" in v:
                out[str(k)] = {"x": float(v["x"]), "y": float(v["y"]), "theta": float(v["theta"])}
        return out
    except (json.JSONDecodeError, OSError):
        return {}


def save_locations(locations: dict[str, LocationCoords], path: Optional[os.PathLike[str] | str] = None) -> None:
    """Write locations to JSON (atomic write)."""
    p = Path(path) if path is not None else _default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(locations, f, indent=2)
    tmp.replace(p)


class LocationStore:
    """Thread-safe in-memory location dict with optional persistence."""

    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        selected_path = Path(path) if path is not None else _default_path()
        self._path = ensure_locations_file(selected_path)
        self._locations: dict[str, LocationCoords] = load_locations(self._path)
        self._lock = threading.Lock()

    def get(self, name: str) -> Optional[LocationCoords]:
        with self._lock:
            return self._locations.get(name)

    def get_all(self) -> dict[str, LocationCoords]:
        with self._lock:
            return dict(self._locations)

    def set(self, name: str, coords: LocationCoords) -> None:
        with self._lock:
            self._locations[name] = coords
            save_locations(self._locations, self._path)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._locations.keys())


class NavigationState:
    """Shared navigation metadata independent of the robot transport."""

    def __init__(self, location_store: LocationStore):
        self.location_store = location_store
        self._last_goal_handle: Any = None
        self._handle_lock = threading.Lock()
        self._active_goal: Optional[dict[str, Any]] = None
        self._dock_alignment_active = False
        self._goal_lock = threading.Lock()
        self._goal_counter = 0
        self._goal_result_delivery: dict[str, str] = {}
        self._goal_result_delivery_limit = 128
        self._interrupted_mission: Optional[dict[str, Any]] = None
        self._return_points: dict[str, LocationCoords] = {}
        self._active_goal_changed_callback: Optional[Callable[[], None]] = None
        self._patrol: dict[str, Any] = {
            "enabled": False,
            "route": [],
            "next_index": 0,
            "awaiting_target": "",
        }
        self._paused_patrol: Optional[dict[str, Any]] = None

    def set_last_goal_handle(self, handle: Any) -> None:
        with self._handle_lock:
            self._last_goal_handle = handle

    def take_last_goal_handle(self) -> Any:
        with self._handle_lock:
            h = self._last_goal_handle
            self._last_goal_handle = None
            return h

    def set_active_goal_changed_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        with self._goal_lock:
            self._active_goal_changed_callback = callback

    def _notify_active_goal_changed(self) -> None:
        callback: Optional[Callable[[], None]]
        with self._goal_lock:
            callback = self._active_goal_changed_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:
            pass

    def new_goal_id(self) -> str:
        """Allocate an Argos navigation goal id before dispatching to a provider."""
        with self._goal_lock:
            self._goal_counter += 1
            return f"nav-{self._goal_counter}"

    def begin_goal(
        self,
        *,
        tool_name: str,
        target_label: str,
        handle: Any = None,
        goal_id: str | None = None,
        waypoint_names: Optional[list[str]] = None,
        policy: NavigationPolicy = INTERRUPTIBLE_NAVIGATION_POLICY,
    ) -> dict[str, Any]:
        """Register a newly-accepted goal and return immutable goal metadata."""
        with self._goal_lock:
            rendered_goal_id = str(goal_id or "").strip()
            if not rendered_goal_id:
                self._goal_counter += 1
                rendered_goal_id = f"nav-{self._goal_counter}"
            self._active_goal = {
                "goal_id": rendered_goal_id,
                "tool_name": tool_name,
                "target_label": target_label,
                "waypoint_names": list(waypoint_names or []),
                "reported_waypoint_indices": set(),
                "policy": policy,
            }
            self._goal_result_delivery[rendered_goal_id] = policy.result_delivery
            while len(self._goal_result_delivery) > self._goal_result_delivery_limit:
                self._goal_result_delivery.pop(next(iter(self._goal_result_delivery)))
            self._last_goal_handle = handle
            # New accepted navigation intent supersedes any previously interrupted one.
            self._interrupted_mission = None
            goal_meta = {
                "goal_id": rendered_goal_id,
                "tool_name": tool_name,
                "target_label": target_label,
                "waypoint_names": list(waypoint_names or []),
                "policy": policy,
            }
        self._notify_active_goal_changed()
        return goal_meta

    def result_delivery_for_goal(self, goal_id: str, *, tool_name: str = "") -> str:
        """Return the single consumer for a completion, including late duplicates."""
        rendered_goal_id = str(goal_id or "").strip()
        rendered_tool_name = str(tool_name or "").strip()
        with self._goal_lock:
            delivery = self._goal_result_delivery.get(rendered_goal_id)
        if delivery in {
            NAV_RESULT_DELIVERY_MODEL_EVENT,
            NAV_RESULT_DELIVERY_TOOL_RESULT,
            NAV_RESULT_DELIVERY_RUNTIME_ONLY,
        }:
            return delivery
        if rendered_tool_name == "patrol_navigation":
            return NAV_RESULT_DELIVERY_RUNTIME_ONLY
        if rendered_tool_name.endswith("_blocking") or rendered_tool_name in {
            "navigation_blocking",
            "charging_dock",
        }:
            return NAV_RESULT_DELIVERY_TOOL_RESULT
        return NAV_RESULT_DELIVERY_MODEL_EVENT

    def is_active_goal(self, goal_id: str) -> bool:
        with self._goal_lock:
            return bool(self._active_goal and self._active_goal.get("goal_id") == goal_id)

    def clear_goal_if_active(self, goal_id: str) -> bool:
        cleared = False
        with self._goal_lock:
            if self._active_goal and self._active_goal.get("goal_id") == goal_id:
                self._active_goal = None
                cleared = True
        if cleared:
            self._notify_active_goal_changed()
        return cleared

    def mark_active_goal_cancel_unconfirmed(self, goal_id: str) -> bool:
        """Keep an uncertain provider goal active and block replacement goals."""
        marked = False
        with self._goal_lock:
            if self._active_goal and self._active_goal.get("goal_id") == goal_id:
                self._active_goal["cancel_unconfirmed"] = True
                marked = True
        if marked:
            self._notify_active_goal_changed()
        return marked

    def has_unconfirmed_active_goal(self) -> bool:
        with self._goal_lock:
            return bool(
                self._active_goal and self._active_goal.get("cancel_unconfirmed", False)
            )

    def begin_dock_alignment(self) -> None:
        with self._goal_lock:
            self._dock_alignment_active = True
        self._notify_active_goal_changed()

    def clear_dock_alignment(self) -> None:
        changed = False
        with self._goal_lock:
            if self._dock_alignment_active:
                self._dock_alignment_active = False
                changed = True
        if changed:
            self._notify_active_goal_changed()

    def has_active_dock_alignment(self) -> bool:
        with self._goal_lock:
            return self._dock_alignment_active

    def get_active_goal(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            if self._active_goal is None:
                return None
            out = dict(self._active_goal)
            reported = out.get("reported_waypoint_indices")
            if isinstance(reported, set):
                out["reported_waypoint_indices"] = set(reported)
            return out

    def set_return_point(self, label: str, coords: LocationCoords) -> None:
        rendered = str(label or "").strip() or "assignment_start"
        with self._goal_lock:
            self._return_points[rendered] = {
                "x": float(coords["x"]),
                "y": float(coords["y"]),
                "theta": float(coords["theta"]),
            }

    def get_return_point(self, label: str) -> Optional[LocationCoords]:
        rendered = str(label or "").strip() or "assignment_start"
        with self._goal_lock:
            coords = self._return_points.get(rendered)
            return dict(coords) if coords is not None else None

    def get_active_policy(self) -> Optional[NavigationPolicy]:
        with self._goal_lock:
            if self._active_goal is None:
                return (
                    CHARGING_DOCK_NAVIGATION_POLICY
                    if self._dock_alignment_active
                    else None
                )
            policy = self._active_goal.get("policy")
            if isinstance(policy, NavigationPolicy):
                return policy
            return INTERRUPTIBLE_NAVIGATION_POLICY

    def active_goal_allows_auto_interrupt(self) -> bool:
        policy = self.get_active_policy()
        if policy is None:
            return False
        return policy.allows_auto_interrupt()

    def allows_proactive_face_attention(self) -> bool:
        policy = self.get_active_policy()
        if policy is None:
            return True
        return policy.allows_proactive_face_attention()

    def active_goal_allows_passive_listen(self) -> bool:
        policy = self.get_active_policy()
        if policy is None:
            return True
        return policy.allows_passive_listen()

    def build_interaction_context(self) -> dict[str, Any]:
        policy = self.get_active_policy()
        if policy is None:
            return {
                "nav_active": False,
                "nav_source": "",
                "nav_interruptible": True,
                "nav_passive_listen_allowed": True,
            }
        return {
            "nav_active": True,
            "nav_source": policy.source,
            "nav_interruptible": policy.interruptible,
            "nav_passive_listen_allowed": policy.passive_listen_allowed,
        }

    def mark_waypoint_reported(self, goal_id: str, index: int) -> bool:
        """Return True when this waypoint index is first reported for active goal."""
        with self._goal_lock:
            if not self._active_goal or self._active_goal.get("goal_id") != goal_id:
                return False
            reported = self._active_goal.get("reported_waypoint_indices")
            if not isinstance(reported, set):
                reported = set()
                self._active_goal["reported_waypoint_indices"] = reported
            if index in reported:
                return False
            reported.add(index)
            return True

    def save_interrupted_mission(self, mission: dict[str, Any]) -> None:
        with self._goal_lock:
            self._interrupted_mission = dict(mission)

    def clear_interrupted_mission(self) -> None:
        with self._goal_lock:
            self._interrupted_mission = None

    def take_interrupted_mission(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            if self._interrupted_mission is None:
                return None
            mission = dict(self._interrupted_mission)
            self._interrupted_mission = None
            return mission

    def peek_interrupted_mission(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            if self._interrupted_mission is None:
                return None
            return dict(self._interrupted_mission)

    def start_patrol(self, route: list[str]) -> None:
        with self._goal_lock:
            first_target = str(route[0]) if route else ""
            self._patrol = {
                "enabled": True,
                "route": list(route),
                "next_index": 1 if len(route) > 1 else 0,
                "awaiting_target": first_target,
            }
            self._paused_patrol = None

    def stop_patrol(self) -> None:
        with self._goal_lock:
            self._patrol = {
                "enabled": False,
                "route": [],
                "next_index": 0,
                "awaiting_target": "",
            }
            self._paused_patrol = None

    def pause_patrol(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            if not self._patrol.get("enabled", False):
                return None
            self._paused_patrol = dict(self._patrol)
            self._patrol = {
                "enabled": False,
                "route": [],
                "next_index": 0,
                "awaiting_target": "",
            }
            return dict(self._paused_patrol)

    def resume_paused_patrol(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            if self._paused_patrol is None:
                return None
            resumed = dict(self._paused_patrol)
            self._patrol = resumed
            self._paused_patrol = None
            return dict(self._patrol)

    def get_patrol(self) -> dict[str, Any]:
        with self._goal_lock:
            return dict(self._patrol)

    def set_patrol_awaiting_target(self, target: str) -> None:
        with self._goal_lock:
            if not self._patrol.get("enabled", False):
                return
            self._patrol["awaiting_target"] = str(target)

    def patrol_mark_arrived_and_get_next(self, arrived_target: str) -> Optional[str]:
        with self._goal_lock:
            if not self._patrol.get("enabled", False):
                return None
            route = list(self._patrol.get("route", []))
            if not route:
                return None
            awaiting = str(self._patrol.get("awaiting_target", "")).strip()
            arrived = str(arrived_target).strip()
            if awaiting and arrived and awaiting != arrived:
                return None
            next_index = int(self._patrol.get("next_index", 0))
            if next_index < 0 or next_index >= len(route):
                next_index = 0
            next_target = route[next_index]
            self._patrol["next_index"] = (next_index + 1) % len(route)
            self._patrol["awaiting_target"] = next_target
            return str(next_target)
