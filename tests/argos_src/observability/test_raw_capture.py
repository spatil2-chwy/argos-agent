from __future__ import annotations

import json
import wave

from argos_src.observability.raw_capture import RawDataCaptureSink


def test_raw_capture_writes_exchange_audio_under_owner_conversation(tmp_path):
    sink = RawDataCaptureSink(tmp_path)
    sink.start_session(run_id="run-test", metadata={"profile": "static_interaction"})

    sink.save_exchange(
        exchange_id="ex-one",
        exchange_index=1,
        owner_id="person-1",
        owner_source="face",
        audio_pcm16=b"\x01\x00\x02\x00",
        sample_rate_hz=16000,
        metadata={"req_id": "rt-one"},
    )
    sink.save_exchange(
        exchange_id="ex-two",
        exchange_index=2,
        owner_id="person-1",
        owner_source="audio_face_agree",
        audio_pcm16=b"\x03\x00",
        sample_rate_hz=16000,
        metadata={"req_id": "rt-two"},
    )
    sink.close()

    session_dir = tmp_path / "run-test"
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert session["run_id"] == "run-test"

    conversations = sorted((session_dir / "conversations").iterdir())
    assert [path.name for path in conversations] == ["conversation-001_owner_person-1"]

    first_exchange = (
        conversations[0] / "exchanges" / "0001_ex-one"
    )
    manifest = json.loads((first_exchange / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["owner_key"] == "owner:person-1"
    assert manifest["audio_file"] == "input_audio_16khz_mono.wav"
    assert manifest["metadata"]["req_id"] == "rt-one"

    with wave.open(str(first_exchange / "input_audio_16khz_mono.wav"), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(2) == b"\x01\x00\x02\x00"

