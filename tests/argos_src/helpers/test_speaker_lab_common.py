from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np

from argos_src.helpers.speaker_lab_common import (
    SAMPLE_RATE,
    build_lab_config,
    inspect_vad_frames,
    load_audio_file_as_agent_pcm16,
    render_frame_rms_payload,
    render_stats_payload,
    session_summary_payload,
    summarize_attempt_diagnostics,
)
from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy


def _write_wav(
    path: Path,
    *,
    audio: np.ndarray,
    sample_rate: int,
    channels: int,
    sample_width: int = 2,
) -> None:
    rendered = np.asarray(audio)
    if channels > 1:
        rendered = rendered.reshape(-1, channels)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(rendered.astype(np.int16).tobytes())


def test_load_audio_file_as_agent_pcm16_converts_to_16k_mono(tmp_path: Path) -> None:
    sample_rate = 8000
    left = np.full(sample_rate, 1000, dtype=np.int16)
    right = np.full(sample_rate, -1000, dtype=np.int16)
    stereo = np.column_stack([left, right]).reshape(-1)
    wav_path = tmp_path / "stereo.wav"
    _write_wav(
        wav_path,
        audio=stereo,
        sample_rate=sample_rate,
        channels=2,
    )

    converted, meta = load_audio_file_as_agent_pcm16(wav_path)

    assert meta["channels"] == 2
    assert meta["sample_rate_hz"] == 8000
    assert meta["converted_to_agent_pcm16"] is True
    assert len(converted) > 0
    assert abs((len(converted) // 2) - SAMPLE_RATE) <= 1


def test_build_lab_config_keeps_speaker_db_isolated_from_agent_db(tmp_path: Path) -> None:
    args = argparse.Namespace(
        profile="static_interaction",
        session_dir=str(tmp_path / "speaker_lab"),
        input_device="",
        input_sample_rate=None,
        input_block_size=None,
        vad_threshold=None,
        silence_grace_period=None,
        listen_timeout_sec=10.0,
        max_record_sec=8.0,
        query_min_voiced_sec=None,
        query_match_threshold=None,
        query_margin_threshold=None,
        reference_update_threshold=None,
        enroll_min_voiced_sec=None,
        enroll_max_voiced_sec=None,
        enroll_min_rms_level=None,
        max_clipped_fraction=None,
    )

    config = build_lab_config(args)
    summary = session_summary_payload(config, vad_impl="test")

    assert config.speaker_db_path == str((tmp_path / "speaker_lab" / "speaker_db").resolve())
    assert config.policy.db_path == config.speaker_db_path
    assert config.profile_speaker_db_path != config.speaker_db_path
    assert summary["speaker_db_isolated_from_agent"] is True


def test_render_stats_payload_reports_expected_duration_and_sample_count() -> None:
    audio = np.full(SAMPLE_RATE * 2, 1200, dtype=np.int16).tobytes()

    payload = render_stats_payload(audio)

    assert payload["sample_rate_hz"] == SAMPLE_RATE
    assert payload["sample_count"] == SAMPLE_RATE * 2
    assert payload["duration_s"] == 2.0
    assert payload["rms_level"] == 1200.0


def test_inspect_vad_frames_counts_voiced_frames() -> None:
    silence = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.int16)
    speech = np.full(int(SAMPLE_RATE * 0.03), 1000, dtype=np.int16)
    waveform = np.concatenate([silence, speech, silence, speech]).tobytes()

    def fake_vad(frame, _context):
        return bool(np.max(np.abs(frame)) >= 999), {}

    payload = inspect_vad_frames(waveform, vad=fake_vad)

    assert payload["total_frames"] == 4
    assert payload["voiced_frames"] == 2
    assert payload["voiced_fraction"] == 0.5
    assert payload["frame_samples"] == int(SAMPLE_RATE * 0.03)


def test_render_frame_rms_payload_summarizes_frame_energy() -> None:
    silence = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.int16)
    speech = np.full(int(SAMPLE_RATE * 0.03), 500, dtype=np.int16)
    waveform = np.concatenate([silence, speech, speech]).tobytes()

    payload = render_frame_rms_payload(waveform)

    assert payload["frame_ms"] == 30
    assert payload["total_frames"] == 3
    assert payload["frames_rms_gt_100"] == 2
    assert payload["frames_rms_gt_400"] == 2


def test_inspect_vad_frames_uses_vad_window_size_when_available() -> None:
    waveform = np.array([0, 0, 0, 0, 1200, 1200, 1200, 1200], dtype=np.int16).tobytes()

    class FakeWindowVad:
        window_size = 4

        def __call__(self, frame, _context):
            return bool(np.max(np.abs(frame)) >= 1000), {}

    payload = inspect_vad_frames(waveform, vad=FakeWindowVad())

    assert payload["frame_samples"] == 4
    assert payload["total_frames"] == 2
    assert payload["voiced_frames"] == 1


def test_summarize_attempt_diagnostics_flags_trim_fallback_and_borderline_match() -> None:
    policy = SpeakerRecognitionPolicy(
        backend="speechbrain_ecapa",
        db_path="/tmp/test_speaker_db",
        query_min_voiced_sec=0.8,
        query_match_threshold=0.6,
        query_margin_threshold=0.08,
        reference_update_threshold=0.55,
        enroll_min_voiced_sec=2.0,
        enroll_max_voiced_sec=0.0,
        enroll_min_rms_level=350.0,
        max_clipped_fraction=0.02,
        explicit_prompt_after_silent_failures=2,
    )
    audio = np.full(SAMPLE_RATE, 300, dtype=np.int16).tobytes()

    payload = summarize_attempt_diagnostics(
        policy=policy,
        vad_impl="rms_fallback",
        raw_audio_pcm16=audio,
        trimmed_audio_pcm16=audio,
        raw_vad_frames={"voiced_frames": 0},
        trimmed_vad_frames={"voiced_frames": 0},
        capture_vad_positive_blocks=3,
        query_safe=True,
        top_score=0.62,
        reference_count=1,
    )

    assert payload["vad_impl"] == "rms_fallback"
    assert payload["kept_ratio"] == 1.0
    assert payload["raw_voiced_frames"] == 0
    assert "using_rms_fallback_vad" in payload["notes"]
    assert "trim_used_raw_audio_fallback_no_voiced_frames" in payload["notes"]
    assert "capture_vad_detected_speech_but_trim_vad_found_none" in payload["notes"]
    assert "clip_quieter_than_enrollment_min_rms" in payload["notes"]
    assert "query_match_is_borderline" in payload["notes"]
    assert "single_reference_match_not_discriminative" in payload["notes"]
