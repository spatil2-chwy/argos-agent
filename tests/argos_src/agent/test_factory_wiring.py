from types import SimpleNamespace

from argos_src.agent.factory_wiring import FactoryRuntimeWireup


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


def test_idle_entered_resumes_patrol_when_idle_and_battery_allows_navigation():
    nav_state = _FakeNavState()
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
        robot_client=SimpleNamespace(),
        nav_state=nav_state,
        battery_cache=battery_cache,
        format_navigation_event=lambda event: str(event),
    )
    wiring.bind_agent(agent)
    wiring.bind_coalescer(coalescer)

    wiring.on_idle_entered()

    assert len(coalescer.submitted) == 1
    assert "PATROL_EVENT" in coalescer.submitted[0][0]
    assert coalescer.submitted[0][1]["target_label"] == "kitchen"
    assert transitions[-1].axis == "robot_arbitration"
    assert transitions[-1].new_state == "patrol_allowed"
    assert transitions[-1].fields["target_label"] == "kitchen"
