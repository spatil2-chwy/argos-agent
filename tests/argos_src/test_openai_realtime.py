from __future__ import annotations

from types import SimpleNamespace

from argos_src.openai_realtime import (
    realtime_audio_session_payload,
    realtime_auth_headers,
    realtime_response_payload,
    realtime_text_session_payload,
    realtime_websocket_url,
)


def test_realtime_websocket_uses_ga_url_and_no_beta_header() -> None:
    assert realtime_websocket_url("gpt-realtime-1.5") == (
        "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5"
    )
    assert realtime_auth_headers("sk-test") == ["Authorization: Bearer sk-test"]


def test_realtime_audio_session_payload_uses_ga_shape() -> None:
    profile = SimpleNamespace(
        model="gpt-realtime-1.5",
        input_sample_rate=24000,
        output_sample_rate=24000,
        noise_reduction="near_field",
        transcription_model="gpt-4o-mini-transcribe",
        language="en",
        voice="cedar",
        audio_output_speed=0.9,
    )

    payload = realtime_audio_session_payload(
        profile=profile,
        instructions="system",
        tools=[{"type": "function", "name": "do_thing"}],
    )

    assert payload["type"] == "realtime"
    assert payload["model"] == "gpt-realtime-1.5"
    assert payload["output_modalities"] == ["audio"]
    assert payload["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "turn_detection": None,
        "noise_reduction": {"type": "near_field"},
        "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
    }
    assert payload["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "cedar",
        "speed": 0.9,
    }
    assert "modalities" not in payload
    assert "input_audio_format" not in payload
    assert "temperature" not in payload


def test_realtime_text_payloads_use_output_modalities() -> None:
    profile = SimpleNamespace(model="gpt-realtime-1.5")

    session = realtime_text_session_payload(
        profile=profile,
        instructions="extract",
        tools=[],
    )
    response = realtime_response_payload(
        instructions="extract",
        output_modalities=["text"],
        max_output_tokens=64,
    )

    assert session["output_modalities"] == ["text"]
    assert "temperature" not in session
    assert response == {
        "instructions": "extract",
        "output_modalities": ["text"],
        "max_output_tokens": 64,
    }
