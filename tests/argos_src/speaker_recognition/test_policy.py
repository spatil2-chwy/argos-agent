import numpy as np

from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy
from argos_src.speaker_recognition.policy import (
    enrollment_rejection_reason,
    is_query_clip_safe,
    resolve_owner_id,
    trim_voice_activity,
)


def _pcm16_with_amplitude(amplitude: int, *, duration_s: float) -> np.ndarray:
    samples = max(1, int(16000 * duration_s))
    return np.full(samples, amplitude, dtype=np.int16)


def test_resolve_owner_id_uses_accepted_audio_above_threshold():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.60, query_margin_threshold=0.08)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id="alice",
        audio_speaker_id="bob",
        top_score=0.61,
        runner_up_score=0.21,
        visible_face_person_ids=("alice",),
    )

    assert result.owner_id == "bob"
    assert result.owner_source == "audio"
    assert result.audio_speaker_id == "bob"
    assert result.speaker_visible is False


def test_resolve_owner_id_falls_back_to_primary_face_when_audio_is_below_threshold():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.60)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id="alice",
        audio_speaker_id="bob",
        top_score=0.515,
        runner_up_score=0.0,
    )

    assert result.owner_id == "alice"
    assert result.owner_source == "face"
    assert result.audio_speaker_id is None


def test_resolve_owner_id_uses_margin_threshold_for_ambiguous_audio():
    policy = SpeakerRecognitionPolicy(
        query_match_threshold=0.60,
        query_margin_threshold=0.08,
    )

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id="alice",
        audio_speaker_id="bob",
        top_score=0.91,
        runner_up_score=0.86,
        visible_face_person_ids=("alice",),
    )

    assert result.owner_id == "alice"
    assert result.audio_speaker_id is None
    assert result.owner_source == "face"


def test_resolve_owner_id_reports_audio_face_agreement_when_accepted_ids_match():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.60, query_margin_threshold=0.08)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id="alice",
        audio_speaker_id="alice",
        top_score=0.61,
        runner_up_score=0.21,
        visible_face_person_ids=("alice",),
    )

    assert result.owner_id == "alice"
    assert result.owner_source == "audio_face_agree"
    assert result.speaker_visible is True


def test_resolve_owner_id_requires_margin_even_when_face_corroborates_audio():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.40, query_margin_threshold=0.20)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id="alice",
        audio_speaker_id="alice",
        top_score=0.61,
        runner_up_score=0.55,
        visible_face_person_ids=("alice",),
    )

    assert result.owner_id == "alice"
    assert result.owner_source == "face"
    assert result.audio_speaker_id is None


def test_resolve_owner_id_leaves_unresolved_without_face_or_accepted_audio():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.60)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id=None,
        audio_speaker_id="alice",
        top_score=0.515,
        runner_up_score=0.0,
    )

    assert result.owner_id is None
    assert result.audio_speaker_id is None
    assert result.owner_source == "unknown"


def test_resolve_owner_id_leaves_unresolved_when_no_face_or_audio_owner():
    policy = SpeakerRecognitionPolicy(query_match_threshold=0.60)

    result = resolve_owner_id(
        policy=policy,
        primary_face_person_id=None,
        audio_speaker_id=None,
        top_score=0.0,
        runner_up_score=0.0,
    )

    assert result.owner_id is None
    assert result.owner_source == "unknown"


def test_enrollment_rejection_reason_rejects_short_audio():
    policy = SpeakerRecognitionPolicy(enroll_min_voiced_sec=2.0)
    waveform = _pcm16_with_amplitude(1200, duration_s=1.0)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == "reject_too_short"


def test_enrollment_rejection_reason_rejects_empty_audio_even_without_duration_gate():
    policy = SpeakerRecognitionPolicy(enroll_min_voiced_sec=0.0, enroll_min_rms_level=0.0)
    waveform = np.asarray([], dtype=np.int16)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == "reject_empty"


def test_enrollment_rejection_reason_rejects_quiet_audio():
    policy = SpeakerRecognitionPolicy(
        enroll_min_voiced_sec=2.0,
        enroll_min_rms_level=350.0,
    )
    waveform = _pcm16_with_amplitude(100, duration_s=2.5)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == "reject_too_quiet"


def test_enrollment_rejection_reason_accepts_clean_clip():
    policy = SpeakerRecognitionPolicy(
        enroll_min_voiced_sec=2.0,
        enroll_min_rms_level=350.0,
        max_clipped_fraction=0.02,
    )
    waveform = _pcm16_with_amplitude(1200, duration_s=2.5)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == ""


def test_enrollment_rejection_reason_allows_clean_audio_without_transcript_dependency():
    policy = SpeakerRecognitionPolicy(
        enroll_min_voiced_sec=2.0,
        enroll_min_rms_level=350.0,
        max_clipped_fraction=0.02,
    )
    waveform = _pcm16_with_amplitude(1200, duration_s=2.5)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == ""


def test_query_clip_safety_depends_only_on_audio():
    policy = SpeakerRecognitionPolicy(query_min_voiced_sec=0.8)
    waveform = _pcm16_with_amplitude(1200, duration_s=1.2)

    assert is_query_clip_safe(
        policy,
        audio_pcm16=waveform,
    ) is True


def test_enrollment_rejection_reason_does_not_need_transcript_to_accept_audio():
    policy = SpeakerRecognitionPolicy(
        enroll_min_voiced_sec=2.0,
        enroll_min_rms_level=350.0,
        max_clipped_fraction=0.02,
    )
    waveform = _pcm16_with_amplitude(1200, duration_s=2.5)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == ""


def test_enrollment_rejection_reason_allows_long_clean_clip_when_max_is_uncapped():
    policy = SpeakerRecognitionPolicy(
        enroll_min_voiced_sec=2.0,
        enroll_max_voiced_sec=0.0,
        enroll_min_rms_level=350.0,
        max_clipped_fraction=0.02,
    )
    waveform = _pcm16_with_amplitude(1200, duration_s=25.0)

    reason = enrollment_rejection_reason(
        policy,
        audio_pcm16=waveform,
    )

    assert reason == ""


def test_trim_voice_activity_collapses_internal_silence_with_vad():
    silence = np.zeros(16000, dtype=np.int16)
    speech = np.full(16000, 1200, dtype=np.int16)
    waveform = np.concatenate([silence, speech, silence, speech, silence])

    def fake_vad(frame, _context):
        voiced = bool(np.max(np.abs(frame)) >= 1000)
        return voiced, {}

    trimmed = trim_voice_activity(waveform, vad=fake_vad)

    frame_samples = int(16000 * 0.03)
    expected_frames_per_span = (speech.shape[0] + frame_samples - 1) // frame_samples
    assert trimmed.shape[0] == expected_frames_per_span * frame_samples * 2
    assert np.max(np.abs(trimmed)) == 1200


def test_trim_voice_activity_honors_vad_window_size_when_available():
    waveform = np.array(
        [0, 0, 0, 0, 1200, 1200, 1200, 1200, 0, 0, 0, 0],
        dtype=np.int16,
    )

    class FakeWindowVad:
        window_size = 4

        def __call__(self, frame, _context):
            return bool(np.max(np.abs(frame)) >= 1000), {}

    trimmed = trim_voice_activity(waveform, vad=FakeWindowVad())

    assert trimmed.shape[0] == 4
    assert np.max(np.abs(trimmed)) == 1200
