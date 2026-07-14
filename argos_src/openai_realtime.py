"""OpenAI Realtime GA request-shape helpers."""

from __future__ import annotations

from typing import Any, Iterable

OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
REALTIME_AUDIO_FORMAT = "audio/pcm"


def realtime_websocket_url(model: str) -> str:
    return f"{OPENAI_REALTIME_WS_URL}?model={str(model or '').strip()}"


def realtime_auth_headers(api_key: str) -> list[str]:
    return [f"Authorization: Bearer {str(api_key or '').strip()}"]


def realtime_audio_session_payload(
    *,
    profile: Any,
    instructions: str,
    tools: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    audio_input: dict[str, Any] = {
        "format": {
            "type": REALTIME_AUDIO_FORMAT,
            "rate": int(getattr(profile, "input_sample_rate", 24000) or 24000),
        },
        "turn_detection": None,
    }
    noise_reduction = str(getattr(profile, "noise_reduction", "") or "").strip()
    if noise_reduction:
        audio_input["noise_reduction"] = {"type": noise_reduction}

    transcription_model = str(getattr(profile, "transcription_model", "") or "").strip()
    if transcription_model:
        transcription: dict[str, Any] = {"model": transcription_model}
        language = str(getattr(profile, "language", "") or "").strip()
        if language:
            transcription["language"] = language
        audio_input["transcription"] = transcription

    return {
        "type": "realtime",
        "model": str(getattr(profile, "model", "") or "").strip(),
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": audio_input,
            "output": {
                "format": {
                    "type": REALTIME_AUDIO_FORMAT,
                    "rate": int(getattr(profile, "output_sample_rate", 24000) or 24000),
                },
                "voice": str(getattr(profile, "voice", "") or "").strip(),
                "speed": float(getattr(profile, "audio_output_speed", 1.0)),
            },
        },
        "tools": list(tools),
        "tool_choice": "auto",
    }


def realtime_text_session_payload(
    *,
    profile: Any,
    instructions: str,
    tools: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "realtime",
        "model": str(getattr(profile, "model", "") or "").strip(),
        "instructions": instructions,
        "output_modalities": ["text"],
        "tools": list(tools),
        "tool_choice": "auto",
    }


def realtime_response_payload(
    *,
    instructions: str,
    output_modalities: Iterable[str],
    max_output_tokens: int | None,
    input_items: Iterable[dict[str, Any]] | None = None,
    conversation: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instructions": instructions,
        "output_modalities": list(output_modalities),
    }
    if conversation:
        payload["conversation"] = conversation
    if input_items is not None:
        payload["input"] = [dict(item) for item in input_items]
    if metadata:
        payload["metadata"] = {
            str(key): str(value)
            for key, value in metadata.items()
            if str(key or "").strip() and value is not None
        }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    return payload
