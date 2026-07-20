from types import SimpleNamespace

from argos_src.agent.factory_wiring import FactoryRuntimeWireup
from argos_src.nav_support.locations import (
    FOCUSED_NAVIGATION_POLICY,
    INTERRUPTIBLE_NAVIGATION_POLICY,
    LocationStore,
    NavigationState,
)
from argos_src.tools.unitree_go2.navigation.toolset import process_navigation_event
from argos_src.provider_api.fake import FakeProviderClient


class _FakeNavState:
    def __init__(self):
        self._patrol = {"enabled": True, "awaiting_target": "kitchen"}
        self._active_goal = None

    def get_patrol(self):
        return dict(self._patrol)

    def get_active_goal(self):
        return self._active_goal


class _FakeCoalescer:
    def __init__(self):
        self.submitted = []

    def submit(self, text, metadata):
        self.submitted.append((text, metadata))


def test_idle_entered_does_not_resume_patrol_when_battery_blocks_navigation():
    nav_state = _FakeNavState()
    coalescer = _FakeCoalescer()
    battery_cache = SimpleNamespace(should_block_general_navigation=lambda: True)
    transitions = []
    agent = SimpleNamespace(
        flush_preference_segments=lambda **_kwargs: None,
        _set_display_mode_async=lambda *_args, **_kwargs: None,
        _state_observer=SimpleNamespace(
            transition=lambda transition: transitions.append(transition)
        ),
    )
    wiring = FactoryRuntimeWireup(
        robot_client=SimpleNamespace(),
        nav_state=nav_state,
        battery_cache=battery_cache,
        format_navigation_event=lambda event: str(event),
    )
    wiring.bind_agent(agent)
    wiring.bind_coalescer(coalescer)

    wiring.on_idle_entered()

    assert coalescer.submitted == []
    assert transitions[-1].axis == "robot_arbitration"
    assert transitions[-1].new_state == "battery_low_blocking"
    assert transitions[-1].reason == "battery_blocks_navigation"


def test_idle_entered_resumes_patrol_directly_when_idle_and_battery_allows_navigation(
    tmp_path,
):
    location_store = LocationStore(tmp_path / "locations.json")
    location_store.set("kitchen", {"x": 1.0, "y": 2.0, "theta": 0.5})
    nav_state = NavigationState(location_store)
    nav_state.start_patrol(["kitchen"])
    robot_client = FakeProviderClient()
    coalescer = _FakeCoalescer()
    battery_cache = SimpleNamespace(should_block_general_navigation=lambda: False)
    transitions = []
    agent = SimpleNamespace(
        flush_preference_segments=lambda **_kwargs: None,
        _set_display_mode_async=lambda *_args, **_kwargs: None,
        _state_observer=SimpleNamespace(
            transition=lambda transition: transitions.append(transition)
        ),
    )
    wiring = FactoryRuntimeWireup(
        robot_client=robot_client,
        nav_state=nav_state,
        battery_cache=battery_cache,
        format_navigation_event=lambda event: str(event),
    )
    wiring.bind_agent(agent)
    wiring.bind_coalescer(coalescer)

    wiring.on_idle_entered()

    assert coalescer.submitted == []
    assert robot_client.navigation_goals[-1]["target_label"] == "kitchen"
    assert robot_client.navigation_goals[-1]["blocking"] is False
    assert robot_client.navigation_goals[-1]["tool_name"] == "patrol_navigation"
    assert transitions[-1].axis == "robot_arbitration"
    assert transitions[-1].new_state == "patrol_allowed"
    assert transitions[-1].fields["target_label"] == "kitchen"


def test_patrol_navigation_events_do_not_reach_model_coalescer():
    nav_state = _FakeNavState()
    coalescer = _FakeCoalescer()
    bridge_events = []
    wiring = FactoryRuntimeWireup(
        robot_client=SimpleNamespace(),
        nav_state=nav_state,
        battery_cache=None,
        format_navigation_event=lambda event: f"NAV_EVENT: {event['target_label']}",
    )
    wiring.bind_coalescer(coalescer)
    wiring.bind_patrol_bridge(SimpleNamespace(on_nav_event=bridge_events.append))

    event = {
        "event_type": "goal_result",
        "outcome": "succeeded",
        "target_label": "kitchen",
        "tool_name": "patrol_navigation",
        "goal_id": "nav-1",
    }

    wiring.submit_nav_event(event)

    assert coalescer.submitted == []
    assert bridge_events == [event]


def test_blocking_navigation_result_is_not_redelivered_as_model_or_patrol_event(tmp_path):
    nav_state = NavigationState(LocationStore(tmp_path / "locations.json"))
    goal = nav_state.begin_goal(
        goal_id="nav-blocking",
        tool_name="navigate_to_location_blocking",
        target_label="dock_door_3",
        policy=FOCUSED_NAVIGATION_POLICY,
    )
    coalescer = _FakeCoalescer()
    bridge_events = []
    wiring = FactoryRuntimeWireup(
        robot_client=SimpleNamespace(),
        nav_state=nav_state,
        battery_cache=None,
        format_navigation_event=lambda event: f"NAV_EVENT: {event['target_label']}",
    )
    wiring.bind_coalescer(coalescer)
    wiring.bind_patrol_bridge(SimpleNamespace(on_nav_event=bridge_events.append))
    event = {
        "event_type": "goal_result",
        "goal_id": goal["goal_id"],
        "outcome": "succeeded",
    }

    process_navigation_event(state=nav_state, event=event, on_nav_event=wiring.submit_nav_event)
    wiring.submit_nav_event(event)

    assert coalescer.submitted == []
    assert bridge_events == []


def test_nonblocking_navigation_result_reaches_only_model_coalescer(tmp_path):
    nav_state = NavigationState(LocationStore(tmp_path / "locations.json"))
    goal = nav_state.begin_goal(
        goal_id="nav-async",
        tool_name="navigate_to_location",
        target_label="lobby",
        policy=INTERRUPTIBLE_NAVIGATION_POLICY,
    )
    coalescer = _FakeCoalescer()
    bridge_events = []
    wiring = FactoryRuntimeWireup(
        robot_client=SimpleNamespace(),
        nav_state=nav_state,
        battery_cache=None,
        format_navigation_event=lambda event: f"NAV_EVENT: {event['target_label']}",
    )
    wiring.bind_coalescer(coalescer)
    wiring.bind_patrol_bridge(SimpleNamespace(on_nav_event=bridge_events.append))

    process_navigation_event(
        state=nav_state,
        event={
            "event_type": "goal_result",
            "goal_id": goal["goal_id"],
            "outcome": "succeeded",
        },
        on_nav_event=wiring.submit_nav_event,
    )

    assert len(coalescer.submitted) == 1
    assert bridge_events == []
