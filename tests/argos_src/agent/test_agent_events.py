from __future__ import annotations

import logging
from types import SimpleNamespace
import threading

from argos_src.agent.agent_events.parsing import (
    server_event_item,
    server_event_item_id,
    server_event_response,
    server_event_response_id,
    server_event_type,
)
from argos_src.agent.control.event_adapter import RealtimeEventAdapter


def test_server_event_parsing_reads_nested_payload_ids():
    event = {
        "type": "response.output_item.done",
        "response": {"id": "resp-1"},
        "item": {"id": "item-1", "response_id": "resp-item"},
    }

    assert server_event_type(event) == "response.output_item.done"
    assert server_event_response(event) == {"id": "resp-1"}
    assert server_event_item(event) == {"id": "item-1", "response_id": "resp-item"}
    assert server_event_response_id(event) == "resp-1"
    assert server_event_item_id(event) == "item-1"


def test_server_event_response_id_falls_back_to_item_or_top_level_fields():
    event = {
        "type": "conversation.item.created",
        "response_id": "resp-top",
        "item_id": "item-top",
    }

    assert server_event_response_id(event) == "resp-top"
    assert server_event_item_id(event) == "item-top"


def test_realtime_event_adapter_routes_ga_event_types_to_agent_handlers():
    calls: list[tuple[str, dict[str, object]]] = []
    agent = SimpleNamespace(
        _session_ready=threading.Event(),
        logger=logging.getLogger("test.argos.agent_events"),
        realtime_profile=SimpleNamespace(model="gpt-realtime", voice="cedar"),
        _session_id="",
        _session_estimated_cost_usd=12.0,
        _handle_output_audio_delta=lambda event: calls.append(("audio", event)),
        _handle_output_transcript_delta=lambda event: calls.append(("transcript", event)),
    )

    adapter = RealtimeEventAdapter(agent)

    assert adapter.handle({"type": "response.output_audio.delta"}) is True
    assert adapter.handle({"type": "response.output_audio_transcript.delta"}) is True
    assert calls == [
        ("audio", {"type": "response.output_audio.delta"}),
        ("transcript", {"type": "response.output_audio_transcript.delta"}),
    ]


def test_realtime_event_adapter_updates_session_state():
    agent = SimpleNamespace(
        _session_ready=threading.Event(),
        logger=logging.getLogger("test.argos.agent_events"),
        realtime_profile=SimpleNamespace(model="gpt-realtime", voice="cedar"),
        _session_id="",
        _session_estimated_cost_usd=12.0,
    )

    adapter = RealtimeEventAdapter(agent)
    created = adapter.handle(
        {
            "type": "session.created",
            "session": {"id": "sess-1", "model": "gpt-realtime"},
        }
    )
    updated = adapter.handle({"type": "session.updated"})

    assert created is True
    assert updated is True
    assert agent._session_id == "sess-1"
    assert agent._session_estimated_cost_usd == 0.0
    assert agent._session_ready.is_set()
