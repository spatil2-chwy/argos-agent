from pathlib import Path

from argos_src.agent.control.tool_runtime import ToolRuntime
from argos_src.nav_support.locations import LocationStore, NavigationState
from argos_src.tools.unitree_go2.navigation.toolset import GetCurrentLocationTool


def test_get_current_location_schema_describes_saving_current_spot(tmp_path: Path) -> None:
    store = LocationStore(tmp_path / "locations.json")
    state = NavigationState(store)

    schema = ToolRuntime.build_schema(
        GetCurrentLocationTool(robot_client=object(), state=state)
    )

    assert schema["name"] == "get_current_location"
    assert "save/remember/mark/name this current spot" in schema["description"]

    properties = schema["parameters"]["properties"]
    assert "Required when save is true" in properties["name"]["description"]
    assert "save, remember, mark, or name" in properties["save"]["description"]
