from __future__ import annotations

from types import SimpleNamespace
import logging
import threading

from argos_src.agent.control.event_adapter import RealtimeEventAdapter


def test_realtime_event_adapter_routes_session_and_audio_events() -> None:
    calls = []
    host = SimpleNamespace(
        logger=logging.getLogger("test.event_adapter"),
        realtime_profile=SimpleNamespace(model="gpt-realtime", voice="cedar"),
        _session_ready=threading.Event(),
        _session_id="",
        _session_estimated_cost_usd=1.0,
        _handle_output_audio_delta=lambda event: calls.append(("audio", event)),
    )
    adapter = RealtimeEventAdapter(host)

    assert adapter.handle(
        {
            "type": "session.created",
            "session": {"id": "sess-1", "model": "gpt-realtime"},
        }
    )
    assert adapter.handle({"type": "session.updated"})
    assert adapter.handle({"type": "response.output_audio.delta", "delta": "AQI="})

    assert host._session_id == "sess-1"
    assert host._session_estimated_cost_usd == 0.0
    assert host._session_ready.is_set()
    assert calls == [("audio", {"type": "response.output_audio.delta", "delta": "AQI="})]
