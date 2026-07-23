from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.labs import biometric_enrollment_lab as lab


class FakeDisplay:
    is_configured = True

    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.messages: list[str] = []
        self.prompts: list[dict] = []
        self.subtitles: list[tuple[str, int]] = []

    def show_message(self, text: str) -> None:
        self.messages.append(text)

    def show_subtitle(self, text: str, *, duration_ms: int = 5000) -> None:
        self.subtitles.append((text, duration_ms))

    def review_text_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return {
            "available": True,
            "accepted": self.accepted,
            "status": "accepted" if self.accepted else "rejected",
        }


class FakeSpeakerBackend:
    def embed_query_clip(self, waveform, *, sample_rate: int):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeSpeakerService:
    policy = object()
    backend = FakeSpeakerBackend()


def test_show_uses_centered_message_not_subtitle() -> None:
    display = FakeDisplay()

    lab._show(display, "Face enrollment", "I will snap 5 photos.")

    assert display.messages == ["Face enrollment\nI will snap 5 photos."]
    assert display.subtitles == []


def test_review_prompt_uses_configured_display() -> None:
    display = FakeDisplay(accepted=True)

    accepted = lab._review_prompt(
        display,
        title="Face enrollment",
        message="I will snap 5 photos.",
        accept_label="Start photos",
        reject_label="Cancel",
    )

    assert accepted is True
    assert display.prompts == [
        {
            "title": "Face enrollment",
            "message": "I will snap 5 photos.",
            "accept_label": "Start photos",
            "reject_label": "Cancel",
            "timeout_sec": 120.0,
        }
    ]


def test_voice_capture_displays_prompt_as_message_and_recording_as_subtitle(
    monkeypatch,
    tmp_path,
) -> None:
    display = FakeDisplay()
    prompt = "This is a test recording for Argos voice enrollment."

    def fake_capture(config, *, vad, on_listening, on_recording_start, on_recording_stop):
        on_listening()
        on_recording_start()
        on_recording_stop("speech complete")
        return {
            "success": True,
            "agent_audio_pcm16": (np.zeros(1600, dtype=np.int16)).tobytes(),
            "source_audio_pcm16": b"",
            "source_sample_rate_hz": 0,
        }

    monkeypatch.setattr(lab, "_capture_microphone_utterance_raw", fake_capture)
    monkeypatch.setattr(lab, "enrollment_rejection_reason", lambda policy, *, audio_pcm16: "")

    result = lab._capture_voice_sample(
        args=SimpleNamespace(voice_countdown_sec=0),
        config=object(),
        vad=object(),
        display=display,
        audio_dir=tmp_path,
        sample_id="voice_0001_attempt_0001",
        prompt=prompt,
        sample_number=1,
        total=5,
        speaker_service=FakeSpeakerService(),
    )

    assert result["accepted"] is True
    assert display.messages[0] == "Voice 1/5\nSilence.\nGet ready."
    assert any(prompt in message for message in display.messages)
    assert any("Start speaking now." in message for message in display.messages)
    assert any(
        message == "Submitting audio 1/5...\nSilence." for message in display.messages
    )
    assert all(lab.VOICE_PROMPT_GUIDANCE not in message for message in display.messages)
    assert display.subtitles == [("Recording audio 1/5", 15000)]
    assert all(prompt not in text for text, _ in display.subtitles)
    assert all(lab.VOICE_PROMPT_GUIDANCE not in text for text, _ in display.subtitles)


def test_consistency_rejects_pairwise_different_identities() -> None:
    with pytest.raises(RuntimeError, match="minimum pairwise similarity"):
        lab._embedding_consistency_summary(
            [
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ],
            threshold=0.6,
        )


def test_identity_resolution_requires_verified_canonical_profile() -> None:
    class UnverifiedIdentity:
        def resolve_identity(self, **kwargs):
            return {
                "success": True,
                "data": {
                    "candidate": {
                        "username": "jdoe",
                        "official_name": "Jane Doe",
                    }
                },
            }

        def get_verified_profile(self, **kwargs):
            return None

    args = SimpleNamespace(
        person_name="Jane Doe",
        username="",
        person_id="",
    )
    with pytest.raises(RuntimeError, match="verified canonical profile"):
        lab._identity_from_args(
            args,
            identity_memory=UnverifiedIdentity(),
            site_code="BOS3",
        )


def test_capture_rejects_too_few_samples_and_removed_commit(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "biometric_enrollment_lab",
            "capture",
            "Jane Doe",
            "--photos",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        lab.main()
    assert "must each be at least 5" in capsys.readouterr().err

    monkeypatch.setattr(
        sys,
        "argv",
        ["biometric_enrollment_lab", "Jane Doe", "--commit"],
    )
    with pytest.raises(SystemExit):
        lab.main()
    assert "--commit was removed" in capsys.readouterr().err


def test_capture_requires_employee_email_prefix(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["biometric_enrollment_lab", "capture", "Jane Doe"],
    )

    with pytest.raises(SystemExit):
        lab.main()

    assert "--username is required for capture" in capsys.readouterr().err


def test_capture_main_is_fully_local(monkeypatch, tmp_path) -> None:
    from dataclasses import dataclass
    import sys

    import scripts.labs.face_lab_common as face_common
    from scripts.labs import biometric_enrollment_commands as commands

    @dataclass
    class FakePolicy:
        min_embedding_similarity: float = 0.6

    class FakeFaceService:
        def shutdown(self):
            return None

    class FakeSpeakerService:
        policy = SimpleNamespace(backend="ecapa", query_match_threshold=0.5)

        def shutdown(self):
            return None

    monkeypatch.setattr(
        lab,
        "load_profile",
        lambda _selection: SimpleNamespace(
            name="test",
            identity_memory=SimpleNamespace(site_code="BOS3"),
        ),
    )
    monkeypatch.setattr(lab, "create_display_runtime_for_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(face_common, "build_enrollment_policy", lambda _args: FakePolicy())
    monkeypatch.setattr(
        face_common,
        "build_face_service",
        lambda _args, *, enrollment_policy: (
            FakeFaceService(),
            {"depth_settings": None, "camera_resource_id": "camera", "robot_client": None},
        ),
    )
    monkeypatch.setattr(lab, "build_lab_config", lambda _args: SimpleNamespace(vad_threshold=0.5))
    monkeypatch.setattr(lab, "build_speaker_service", lambda _config: FakeSpeakerService())
    monkeypatch.setattr(lab, "build_vad", lambda _threshold: (object(), "fake"))
    monkeypatch.setattr(lab, "session_summary_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(lab, "write_session_manifest", lambda **kwargs: None)
    samples = [
        {"embedding": np.asarray([1.0, 0.0], dtype=np.float32)},
        {"embedding": np.asarray([0.9, 0.1], dtype=np.float32)},
        {"embedding": np.asarray([1.0, 0.05], dtype=np.float32)},
        {"embedding": np.asarray([0.95, 0.05], dtype=np.float32)},
        {"embedding": np.asarray([1.0, 0.1], dtype=np.float32)},
    ]
    monkeypatch.setattr(lab, "_collect_face_samples", lambda **kwargs: list(samples))
    monkeypatch.setattr(lab, "_collect_voice_samples", lambda **kwargs: list(samples))
    monkeypatch.setattr(
        commands,
        "create_identity_memory_client_for_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("capture must not construct a Tailwag client")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "biometric_enrollment_lab",
            "capture",
            "Jane Doe",
            "--username",
            "jdoe",
            "--output-root",
            str(tmp_path),
            "--no-display",
        ],
    )

    assert lab.main() == 0
    bundles = lab.create_bundle.__module__
    assert bundles == "scripts.labs.biometric_enrollment_bundle"
