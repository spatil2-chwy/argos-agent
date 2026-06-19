#!/usr/bin/env python3
"""Generate one-off OpenAI speech without starting the Argos agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request
import wave

import numpy as np

from argos_src.profile_config import REPO_ROOT

_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "cedar"
DEFAULT_FORMAT = "wav"
DEFAULT_INSTRUCTIONS = "Speak naturally and clearly."
DEFAULT_OUTPUT_DEVICE = "pipewire"
SUPPORTED_FORMATS = ("wav", "mp3", "flac", "opus", "aac", "pcm")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make an OpenAI voice say specific text without launching the "
            "Realtime agent."
        )
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to speak. Multiple words are joined with spaces.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Read the text to speak from a UTF-8 file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Speech model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice name or custom voice id (default: {DEFAULT_VOICE}).",
    )
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="Optional voice direction, such as tone, accent, or pacing.",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=DEFAULT_FORMAT,
        help=f"Output audio format (default: {DEFAULT_FORMAT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the audio file. Defaults under var/labs/openai_say/.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated WAV through the local speaker after saving it.",
    )
    parser.add_argument(
        "--output-device",
        default=DEFAULT_OUTPUT_DEVICE,
        help=(
            "Optional sounddevice output device name or index for --play "
            f"(default: {DEFAULT_OUTPUT_DEVICE})."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List local audio devices and exit.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60).",
    )
    return parser.parse_args()


def resolve_text(args: argparse.Namespace) -> str:
    if args.input_file is not None and args.text:
        raise ValueError("Pass either text arguments or --input-file, not both.")

    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8").strip()

    if args.text:
        return " ".join(args.text).strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return ""


def default_output_path(audio_format: str, now: datetime | None = None) -> Path:
    rendered_now = now or datetime.now()
    timestamp = rendered_now.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "var" / "labs" / "openai_say" / f"say_{timestamp}.{audio_format}"


def build_speech_payload(
    *,
    model: str,
    voice: str,
    text: str,
    instructions: str,
    audio_format: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.strip(),
        "voice": voice.strip(),
        "input": text,
        "response_format": audio_format.strip(),
    }
    rendered_instructions = instructions.strip()
    if rendered_instructions:
        payload["instructions"] = rendered_instructions
    return payload


def request_speech(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _SPEECH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=timeout_seconds) as response:
        return response.read()


def write_audio_file(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)


def resolve_output_device(value: str | None) -> str | int | None:
    if value is None:
        return None

    rendered = str(value).strip()
    if not rendered:
        return None
    if rendered.isdigit():
        return int(rendered)
    return rendered


def list_audio_devices() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on local audio stack.
        raise RuntimeError(f"Could not import sounddevice: {exc}") from exc

    print(sd.query_devices())


def play_wav(path: Path, output_device: str | int | None = None) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on local audio stack.
        raise RuntimeError(f"Could not import sounddevice: {exc}") from exc

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"Can only play 16-bit PCM WAV locally; got sample_width={sample_width}."
        )

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()


def main() -> int:
    args = parse_arguments()

    if args.list_devices:
        try:
            list_audio_devices()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    try:
        text = resolve_text(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not text:
        print(
            "No text provided. Pass text arguments, --input-file, or pipe text on stdin.",
            file=sys.stderr,
        )
        return 2

    if args.play and args.format != "wav":
        print("--play requires --format wav so the lab can decode it locally.", file=sys.stderr)
        return 2

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "OPENAI_API_KEY is not set in the current shell.\n"
            "Run 'source setup_shell.sh' if your key is loaded there, or export it first.",
            file=sys.stderr,
        )
        return 2

    payload = build_speech_payload(
        model=args.model,
        voice=args.voice,
        text=text,
        instructions=args.instructions,
        audio_format=args.format,
    )
    output_path = (args.output or default_output_path(args.format)).resolve()

    try:
        audio = request_speech(
            api_key=api_key,
            payload=payload,
            timeout_seconds=args.timeout,
        )
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        print(f"OpenAI speech request failed: HTTP {exc.code}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"OpenAI speech request failed: {exc.reason}", file=sys.stderr)
        return 1

    write_audio_file(output_path, audio)
    print(f"Wrote {output_path}")

    if args.play:
        try:
            play_wav(
                output_path,
                output_device=resolve_output_device(args.output_device),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Could not play audio: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
