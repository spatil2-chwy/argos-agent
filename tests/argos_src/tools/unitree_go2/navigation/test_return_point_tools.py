from __future__ import annotations

import json
from pathlib import Path

from argos_src.agent.control.tool_runtime import ToolRuntime
from argos_src.nav_support.locations import LocationStore, NavigationState
from argos_src.provider_api.fake import FakeProviderClient
from argos_src.tools.unitree_go2.navigation.toolset import (
    LocalizeCurrentLocationTool,
    MarkReturnPointTool,
    NavigateToReturnPointBlockingTool,
    SaveCurrentLocationTool,
)


def _state(tmp_path: Path) -> NavigationState:
    return NavigationState(LocationStore(tmp_path / "locations.json"))


def test_return_point_tools_mark_and_navigate_back(tmp_path: Path) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)

    mark_result = json.loads(
        MarkReturnPointTool(robot_client=robot, state=state)._run("assignment_start")
    )

    assert mark_result["success"] is True
    assert mark_result["label"] == "assignment_start"
    assert state.get_return_point("assignment_start") == {
        "x": 0.0,
        "y": 0.0,
        "theta": 0.0,
    }

    nav_result = json.loads(
        NavigateToReturnPointBlockingTool(robot_client=robot, state=state)._run(
            "assignment_start"
        )
    )

    assert nav_result["success"] is True
    assert nav_result["status"] == "completed"
    assert robot.navigation_goals[-1]["blocking"] is True
    assert robot.navigation_goals[-1]["tool_name"] == "navigate_to_return_point_blocking"
    assert robot.navigation_goals[-1]["target_label"] == "return_point:assignment_start"


def test_return_point_navigation_requires_marked_point(tmp_path: Path) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)

    result = json.loads(
        NavigateToReturnPointBlockingTool(robot_client=robot, state=state)._run(
            "assignment_start"
        )
    )

    assert result["success"] is False
    assert result["status"] == "error"
    assert "not marked" in result["message"]
    assert robot.navigation_goals == []


def test_save_current_location_persists_named_location(tmp_path: Path) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)

    result = json.loads(
        SaveCurrentLocationTool(robot_client=robot, state=state)._run("office_corner")
    )

    assert result["success"] is True
    assert state.location_store.get("office_corner") == {
        "x": 0.0,
        "y": 0.0,
        "theta": 0.0,
    }


def test_localize_current_location_reports_no_saved_locations(tmp_path: Path) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)

    result = json.loads(LocalizeCurrentLocationTool(robot_client=robot, state=state)._run())

    assert result["success"] is True
    assert result["confidence"] == "unknown"
    assert result["nearest_location"] == ""
    assert result["saved_location_count"] == 0
    assert result["pose"] == {"x": 0.0, "y": 0.0, "theta": 0.0}


def test_localize_current_location_reports_nearest_saved_location(
    tmp_path: Path,
) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)
    state.location_store.set("office_desk", {"x": 0.5, "y": 0.0, "theta": 0.0})
    state.location_store.set("plants", {"x": 5.0, "y": 0.0, "theta": 0.0})

    result = json.loads(LocalizeCurrentLocationTool(robot_client=robot, state=state)._run())

    assert result["success"] is True
    assert result["confidence"] == "near"
    assert result["nearest_location"] == "office_desk"
    assert result["distance_m"] == 0.5
    assert result["saved_location_count"] == 2


def test_localize_current_location_reports_unknown_when_far_from_saved_locations(
    tmp_path: Path,
) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)
    state.location_store.set("plants", {"x": 5.0, "y": 0.0, "theta": 0.0})

    result = json.loads(LocalizeCurrentLocationTool(robot_client=robot, state=state)._run())

    assert result["success"] is True
    assert result["confidence"] == "unknown"
    assert result["nearest_location"] == "plants"
    assert result["distance_m"] == 5.0


def test_new_navigation_tool_schemas_describe_intended_use(tmp_path: Path) -> None:
    robot = FakeProviderClient()
    state = _state(tmp_path)

    localize_schema = ToolRuntime.build_schema(
        LocalizeCurrentLocationTool(robot_client=robot, state=state)
    )
    mark_schema = ToolRuntime.build_schema(
        MarkReturnPointTool(robot_client=robot, state=state)
    )
    return_schema = ToolRuntime.build_schema(
        NavigateToReturnPointBlockingTool(robot_client=robot, state=state)
    )
    save_schema = ToolRuntime.build_schema(
        SaveCurrentLocationTool(robot_client=robot, state=state)
    )

    assert localize_schema["name"] == "localize_current_location"
    assert "read-only" in localize_schema["description"]
    assert mark_schema["name"] == "mark_return_point"
    assert "temporary return point" in mark_schema["description"]
    assert return_schema["name"] == "navigate_to_return_point_blocking"
    assert "final spoken report" in return_schema["description"]
    assert save_schema["name"] == "save_current_location"
    assert "Persist" in save_schema["description"]
