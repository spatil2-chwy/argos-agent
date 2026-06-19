from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from scripts.labs.openai_say_lab import (
    build_speech_payload,
    default_output_path,
    resolve_text,
)


def test_resolve_text_joins_positional_words() -> None:
    args = argparse.Namespace(text=["hello", "there"], input_file=None)

    assert resolve_text(args) == "hello there"


def test_resolve_text_rejects_text_and_input_file(tmp_path: Path) -> None:
    input_path = tmp_path / "say.txt"
    input_path.write_text("hello", encoding="utf-8")
    args = argparse.Namespace(text=["hi"], input_file=input_path)

    try:
        resolve_text(args)
    except ValueError as exc:
        assert "either text arguments or --input-file" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_speech_payload_omits_empty_instructions() -> None:
    payload = build_speech_payload(
        model=" gpt-4o-mini-tts ",
        voice=" cedar ",
        text="testing one two",
        instructions=" ",
        audio_format=" wav ",
    )

    assert payload == {
        "model": "gpt-4o-mini-tts",
        "voice": "cedar",
        "input": "testing one two",
        "response_format": "wav",
    }


def test_default_output_path_uses_lab_var_directory() -> None:
    path = default_output_path(
        "wav",
        now=datetime(2026, 6, 19, 13, 45, 30),
    )

    assert path.name == "say_20260619_134530.wav"
    assert path.parts[-4:] == ("var", "labs", "openai_say", path.name)
