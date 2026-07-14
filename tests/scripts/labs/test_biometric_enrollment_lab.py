from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.labs import biometric_enrollment_lab as lab


class FakeIdentityMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def enroll_face_reference(self, **kwargs):
        self.calls.append(("enroll_face", kwargs))
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": kwargs["person_id"],
            "reference_id": "face-ref",
        }

    def enroll_voice_reference(self, **kwargs):
        self.calls.append(("enroll_voice", kwargs))
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": kwargs["person_id"],
            "reference_id": "voice-ref",
        }

    def observe_face_embedding(self, **kwargs):
        self.calls.append(("observe_face", kwargs))
        return {
            "accepted": True,
            "status": "updated",
            "reason": "updated",
            "person_id": kwargs["person_id"],
            "reference_id": "face-ref",
            "modality": "face",
            "sample_count": len([call for call in self.calls if call[0] == "observe_face"]) + 1,
            "target_sample_count": 5,
        }

    def observe_voice_embedding(self, **kwargs):
        self.calls.append(("observe_voice", kwargs))
        return {
            "accepted": True,
            "status": "updated",
            "reason": "updated",
            "person_id": kwargs["person_id"],
            "reference_id": "voice-ref",
            "modality": "voice",
            "sample_count": len([call for call in self.calls if call[0] == "observe_voice"]) + 1,
            "target_sample_count": 5,
        }


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


def test_commit_modality_enrolls_once_then_observes_remaining_face_samples() -> None:
    fake = FakeIdentityMemory()
    embeddings = [np.asarray([1.0, float(index)], dtype=np.float32) for index in range(5)]

    result = lab._commit_modality(
        identity_memory=fake,
        modality="face",
        person_id="person_jane",
        embeddings=embeddings,
        metadata={"display_name": "Jane Doe"},
    )

    assert [name for name, _ in fake.calls] == [
        "enroll_face",
        "observe_face",
        "observe_face",
        "observe_face",
        "observe_face",
    ]
    assert result["enrollment"]["saved"] is True
    assert len(result["updates"]) == 4
    assert fake.calls[0][1]["consent_status"] == "consented"
    assert fake.calls[1][1]["evidence"] == lab._operator_enrollment_evidence("person_jane")
    assert fake.calls[1][1]["metadata"]["enrollment_mode"] == "operator_controlled_live"


def test_commit_modality_enrolls_once_then_observes_remaining_voice_samples() -> None:
    fake = FakeIdentityMemory()
    embeddings = [np.asarray([1.0, float(index)], dtype=np.float32) for index in range(5)]

    result = lab._commit_modality(
        identity_memory=fake,
        modality="voice",
        person_id="person_jane",
        embeddings=embeddings,
        metadata={"display_name": "Jane Doe"},
    )

    assert [name for name, _ in fake.calls] == [
        "enroll_voice",
        "observe_voice",
        "observe_voice",
        "observe_voice",
        "observe_voice",
    ]
    assert result["enrollment"]["saved"] is True
    assert len(result["updates"]) == 4
    assert fake.calls[0][1]["consent_status"] == "consented"
    assert fake.calls[1][1]["evidence"] == lab._operator_enrollment_evidence("person_jane")
    assert fake.calls[1][1]["evidence"]["audio_speaker_id"] == "person_jane"


def test_slug_person_id_prefers_username_then_email_then_name() -> None:
    assert lab._slug_person_id("Jane Doe", {"username": "jdoe"}) == "person_jdoe"
    assert lab._slug_person_id("Jane Doe", {"employee_email": "jane@example.com"}) == "person_jane"
    assert lab._slug_person_id("Jane A. Doe", {}) == "person_jane_a_doe"
