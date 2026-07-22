import pytest

from argos_src.tools.tool_ids import (
    required_capability_ids_for_tool_id,
    resolve_builtin_tool_name,
    resolve_builtin_tool_names,
)


def test_capability_style_tool_ids_resolve_for_go2():
    assert resolve_builtin_tool_name(
        "posture.stand",
        robot_family="unitree_go2",
    ) == "go2_balance_stand"
    assert resolve_builtin_tool_name(
        "motion.move_robot",
        robot_family="unitree_go2",
    ) == "move_robot"
    assert resolve_builtin_tool_name(
        "embodiment.unitree_go2.hello",
        robot_family="unitree_go2",
    ) == "go2_hello"
    assert resolve_builtin_tool_name(
        "memory.search_semantic",
        robot_family="unitree_go2",
    ) == "search_memory_semantic"
    assert resolve_builtin_tool_name(
        "navigation.localize_current_location",
        robot_family="unitree_go2",
    ) == "localize_current_location"
    assert resolve_builtin_tool_name(
        "navigation.mark_return_point",
        robot_family="unitree_go2",
    ) == "mark_return_point"
    assert resolve_builtin_tool_name(
        "navigation.navigate_to_return_point_blocking",
        robot_family="unitree_go2",
    ) == "navigate_to_return_point_blocking"
    assert resolve_builtin_tool_name(
        "navigation.save_current_location",
        robot_family="unitree_go2",
    ) == "save_current_location"


def test_capability_style_posture_ids_resolve_for_spot():
    assert resolve_builtin_tool_name("posture.stand", robot_family="spot") == "spot_stand"
    assert resolve_builtin_tool_name("posture.self_right", robot_family="spot") == (
        "spot_self_right"
    )
    assert resolve_builtin_tool_name(
        "memory.search_semantic",
        robot_family="spot",
    ) == "search_memory_semantic"


def test_old_tool_ids_are_rejected():
    with pytest.raises(ValueError, match="Unknown built-in tool id"):
        resolve_builtin_tool_names(
            [
                "unitree_go2.actions.go2_hello",
                "unitree_go2.locomotion.move_robot",
                "navigation.get_current_location",
            ],
            robot_family="unitree_go2",
        )


def test_generic_tool_id_without_family_rejects_ambiguous_posture():
    with pytest.raises(ValueError, match="robot_family is required"):
        resolve_builtin_tool_name("posture.stand")


def test_required_capabilities_for_common_tool_ids():
    assert required_capability_ids_for_tool_id(
        "motion.move_robot",
        robot_family="unitree_go2",
    ) == ("motion.velocity",)
    assert required_capability_ids_for_tool_id(
        "vision.capture_scene",
        robot_family="unitree_go2",
    ) == ("camera.rgb",)
    assert required_capability_ids_for_tool_id(
        "posture.stand",
        robot_family="unitree_go2",
    ) == ("posture.command",)
    assert required_capability_ids_for_tool_id(
        "memory.search_semantic",
        robot_family="unitree_go2",
    ) == ()
    assert required_capability_ids_for_tool_id(
        "navigation.navigate_to_location_blocking",
        robot_family="unitree_go2",
    ) == ("navigation.goal", "transform.lookup")
    assert required_capability_ids_for_tool_id(
        "navigation.localize_current_location",
        robot_family="unitree_go2",
    ) == ("navigation.goal", "transform.lookup")
    assert required_capability_ids_for_tool_id(
        "navigation.mark_return_point",
        robot_family="unitree_go2",
    ) == ("navigation.goal", "transform.lookup")
    assert required_capability_ids_for_tool_id(
        "navigation.navigate_to_return_point_blocking",
        robot_family="unitree_go2",
    ) == ("navigation.goal", "transform.lookup")
    assert required_capability_ids_for_tool_id(
        "navigation.save_current_location",
        robot_family="unitree_go2",
    ) == ("navigation.goal", "transform.lookup")
    assert required_capability_ids_for_tool_id(
        "dock.charging",
        robot_family="unitree_go2",
    ) == ("dock.final_alignment", "navigation.goal", "transform.lookup")
