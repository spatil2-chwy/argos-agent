#!/usr/bin/env python3
"""Capture biometric enrollment bundles locally and push them with approval.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.biometric_enrollment_lab capture "Jane Doe"
    poetry run python -m scripts.labs.biometric_enrollment_lab list
    poetry run python -m scripts.labs.biometric_enrollment_lab push

Capture performs no Tailwag requests. It stores accepted raw media and one
locally aggregated face and voice embedding. The push command verifies identity,
checks which Tailwag references already exist, and uploads only missing
modalities after typed operator approval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.identity_memory import TailwagHttpIdentityMemoryClient
from argos_src.speaker_recognition.policy import SAMPLE_RATE, clip_stats, enrollment_rejection_reason
from scripts.labs.biometric_enrollment_bundle import (
    BUNDLE_MANIFEST_FILENAME,
    atomic_write_json,
    create_bundle,
    finalize_bundle,
    normalize_mean_embedding,
)
from scripts.labs.biometric_enrollment_commands import (
    cleanup_local_bundle,
    list_local_bundles,
    normalize_cli_argv,
    push_local_bundle,
)
from scripts.labs.enrollment_audio_collection import _capture_microphone_utterance_raw
from scripts.labs.enrollment_collection_common import (
    DEFAULT_COLLECTION_ROOT,
    create_display_runtime_for_profile,
    json_ready,
    load_profile,
    write_session_manifest,
)
from scripts.labs.perception_lab_common import append_jsonl, write_json
from scripts.labs.speaker_lab_common import (
    build_lab_config,
    build_speaker_service,
    build_vad,
    configure_logging,
    render_stats_payload,
    session_summary_payload,
    write_pcm16_wav,
)


logger = logging.getLogger(__name__)

PHOTO_GUIDANCE = (
    "Face the camera straight on.",
    "Turn your head slightly left.",
    "Turn your head slightly right.",
    "Smile naturally.",
    "Move a little closer and keep your whole face in frame.",
)

VOICE_PROMPTS = (
    "This is a test recording for voice enrollment.",
    "It's weird to be recording my voice for this.",
    "Since when did I start talking to robots?",
    "Okay almost done with the enrollment.",
    "This is the final sentence I am recording.",
)

VOICE_PROMPT_GUIDANCE = "Say what you see on screen, or say any sentence you want."
VOICE_COUNTDOWN_DETAIL = "Silence.\nGet ready."
VOICE_LISTENING_DETAIL = "Start speaking now."
VOICE_SUBMITTING_DETAIL = "Silence."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture biometrics locally, then list, push, or clean completed bundles.",
        usage="%(prog)s {capture PERSON_NAME|list|push|cleanup} [options]",
        epilog=(
            "A legacy name-only capture is still accepted. --commit was removed; "
            "run capture first, then run push for an approved upload."
        ),
    )
    parser.add_argument(
        "person_name",
        nargs="?",
        default="",
        help="Official first and last name for a local capture.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="List local biometric bundles.")
    action.add_argument("--push", action="store_true", help="Choose and push a ready bundle.")
    action.add_argument(
        "--cleanup", action="store_true", help="Choose and delete a completed bundle."
    )
    parser.add_argument("--site-code", default="", help="Tailwag/Snowflake site code, e.g. BOS3.")
    parser.add_argument("--username", default="", help="Verified directory username, if known.")
    parser.add_argument(
        "--person-id",
        default="",
        help="Claimed existing Tailwag person id; push verifies its profile and name.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-start capture phases. This never approves a Tailwag push.",
    )
    parser.add_argument("--photos", type=int, default=5, help="Accepted photo samples required.")
    parser.add_argument("--voice-clips", type=int, default=5, help="Accepted voice clips required.")
    parser.add_argument("--max-photo-attempts", type=int, default=12)
    parser.add_argument("--max-voice-attempts", type=int, default=10)
    parser.add_argument("--photo-countdown-sec", type=int, default=5)
    parser.add_argument("--voice-countdown-sec", type=int, default=5)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_COLLECTION_ROOT),
        help="Root directory for collected artifacts. Default: data_collection.",
    )
    parser.add_argument("--input-device", default="")
    parser.add_argument("--input-sample-rate", type=int, default=None)
    parser.add_argument("--input-block-size", type=int, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--silence-grace-period", type=float, default=None)
    parser.add_argument("--listen-timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-record-sec", type=float, default=8.0)
    parser.add_argument("--query-match-threshold", type=float, default=None)
    parser.add_argument("--query-margin-threshold", type=float, default=None)
    parser.add_argument("--max-clipped-fraction", type=float, default=None)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not update the configured interaction display.",
    )
    _add_face_profile_args(parser)
    _add_enrollment_policy_args(parser)
    return parser


def _add_face_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="static_interaction",
        help="Argos profile name or YAML path. Default: static_interaction.",
    )
    parser.add_argument(
        "--camera-resource",
        default="",
        help="Override resources.face_camera.",
    )
    parser.add_argument(
        "--provider-transport",
        default="",
        help="Override the profile provider transport for lab runs. Use 'fake' for smoke tests.",
    )
    parser.add_argument("--disable-depth", action="store_true")
    parser.add_argument("--sync-slop-sec", type=float, default=None)
    parser.add_argument("--sync-queue-size", type=int, default=None)
    parser.add_argument("--capture-timeout-sec", type=float, default=None)
    parser.add_argument("--max-face-depth-m", type=float, default=None)
    parser.add_argument("--min-valid-samples", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--search-radius-px", type=int, default=None)
    parser.add_argument("--max-valid-depth-m", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")


def _add_enrollment_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-face-area", type=int, default=None)
    parser.add_argument("--min-brightness", type=float, default=None)
    parser.add_argument("--max-brightness", type=float, default=None)
    parser.add_argument("--min-contrast", type=float, default=None)
    parser.add_argument("--min-embedding-similarity", type=float, default=None)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _slug_person_id(name: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = dict(metadata or {})
    username = str(metadata.get("username") or "").strip().lower()
    if username:
        return f"person_{username}"
    email = str(metadata.get("employee_email") or metadata.get("email") or "").strip().lower()
    if email and "@" in email:
        return f"person_{email.split('@', 1)[0]}"
    slug = "_".join(
        part
        for part in re.sub(r"[^a-zA-Z0-9]+", " ", str(name or "").casefold()).split()
        if part
    )
    return f"person_{slug}" if slug else "person_unknown"


def _split_name(name: str) -> tuple[str, str]:
    parts = [part for part in str(name or "").strip().split() if part]
    if len(parts) < 2:
        return (parts[0] if parts else "", "")
    return parts[0], " ".join(parts[1:])


def _confirm(prompt: str, *, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"{prompt} [auto-yes]")
        return True
    answer = input(f"{prompt} [y/N] ").strip().casefold()
    return answer in {"y", "yes"}


def _review_prompt(
    display: Any | None,
    *,
    title: str,
    message: str,
    accept_label: str = "Accept",
    reject_label: str = "Reject",
    assume_yes: bool = False,
) -> bool:
    if assume_yes:
        print(f"{title}: {message} [auto-yes]")
        return True
    if display is not None and bool(getattr(display, "is_configured", False)):
        review = display.review_text_prompt(
            title=title,
            message=message,
            accept_label=accept_label,
            reject_label=reject_label,
            timeout_sec=120.0,
        )
        if not bool(review.get("available", False)):
            raise RuntimeError(f"Display prompt unavailable: {review.get('status')}")
        return bool(review.get("accepted", False))
    return _confirm(f"{title}: {message}", assume_yes=False)


def _show(display: Any | None, message: str, detail: str = "") -> None:
    if display is None:
        return
    try:
        lines = [str(message or "").strip(), str(detail or "").strip()]
        display.show_message("\n".join(line for line in lines if line))
    except Exception:
        logger.debug("Display update failed", exc_info=True)


def _show_subtitle(display: Any | None, text: str, *, duration_ms: int = 15000) -> None:
    if display is None:
        return
    try:
        display.show_subtitle(text, duration_ms=duration_ms)
    except Exception:
        logger.debug("Display subtitle failed", exc_info=True)


def _discard_sample_artifacts(sample: dict[str, Any], *, session_dir: Path) -> None:
    root = session_dir.resolve()
    for raw_path in dict(sample.get("artifacts") or {}).values():
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            logger.warning("Refusing to delete sample artifact outside bundle: %s", path)
            continue
        if path.is_file():
            path.unlink()


def _voice_sample_title(sample_number: int, total: int) -> str:
    return f"Voice {sample_number}/{total}"


def _voice_sentence_detail(prompt: str) -> str:
    return f"{VOICE_LISTENING_DETAIL}\n\n{str(prompt or '').strip()}"


def _countdown(display: Any | None, seconds: int) -> None:
    bounded = max(0, int(seconds))
    if display is not None and bounded > 0:
        try:
            display.show_countdown(bounded)
        except Exception:
            logger.debug("Display countdown failed", exc_info=True)
    if bounded > 0:
        time.sleep(float(bounded))


def _identity_from_args(
    args: argparse.Namespace,
    *,
    identity_memory: TailwagHttpIdentityMemoryClient,
    site_code: str,
) -> dict[str, Any]:
    person_name = str(args.person_name or "").strip()
    username = str(args.username or "").strip().lower()
    explicit_person_id = str(args.person_id or "").strip()
    if explicit_person_id:
        return {
            "person_id": explicit_person_id,
            "official_name": person_name,
            "display_name": person_name,
            "username": username,
            "site_code": site_code,
            "resolution": {"status": "person_id_override"},
        }

    if username:
        verified = identity_memory.get_verified_profile(
            username=username,
            official_name=person_name,
        )
        if verified:
            payload = _plain(verified)
            payload["display_name"] = payload.get("official_name") or person_name
            payload["site_code"] = site_code
            payload["resolution"] = {"status": "verified_profile"}
            return payload
        if not args.allow_unresolved:
            raise RuntimeError(
                f"Could not verify username={username!r} with official_name={person_name!r}. "
                "Use --person-id or --allow-unresolved only if this is intentional."
            )

    first_name, last_name = _split_name(person_name)
    resolution = identity_memory.resolve_identity(
        shared_first_name=first_name,
        shared_last_name=last_name,
        shared_name=person_name,
    )
    resolution_payload = _plain(resolution)
    if bool(resolution_payload.get("success")):
        candidate = dict((resolution_payload.get("data") or {}).get("candidate") or {})
        candidate_username = str(candidate.get("username") or "").strip().lower()
        candidate_name = str(candidate.get("official_name") or person_name).strip()
        verified = identity_memory.get_verified_profile(
            username=candidate_username,
            official_name=candidate_name,
        )
        if verified:
            payload = _plain(verified)
            payload["display_name"] = payload.get("official_name") or candidate_name
            payload["site_code"] = site_code
            payload["resolution"] = resolution_payload
            return payload
        if not args.allow_unresolved:
            raise RuntimeError(
                "Directory resolution succeeded, but Tailwag did not return a "
                "verified canonical profile."
            )
        metadata = {**candidate, "username": candidate_username, "site_code": site_code}
        return {
            **metadata,
            "person_id": _slug_person_id(candidate_name, metadata),
            "official_name": candidate_name,
            "display_name": candidate_name,
            "resolution": resolution_payload,
        }

    if not args.allow_unresolved:
        raise RuntimeError(
            "Directory resolution did not produce a single verified person: "
            f"{json.dumps(resolution_payload, sort_keys=True, default=str)}. "
            "Use --person-id or --allow-unresolved only if this is intentional."
        )
    metadata = {"username": username, "site_code": site_code}
    return {
        **metadata,
        "person_id": _slug_person_id(person_name, metadata),
        "official_name": person_name,
        "display_name": person_name,
        "resolution": resolution_payload,
    }


def _capture_face_sample(
    *,
    service: Any,
    camera_resource_id: str,
    timeout_sec: float,
    output_dir: Path,
    sample_id: str,
    person_id: str,
    check_remote_match: bool = False,
) -> dict[str, Any]:
    from scripts.labs.face_lab_common import (
        describe_enrollment_face_quality,
        save_preview_image,
        summarize_face,
    )

    image, depth_m = service._capture_for_recognition(camera_resource_id, timeout=timeout_sec)
    if image is None:
        return {
            "sample_id": sample_id,
            "accepted": False,
            "reason": "capture_failed",
            "message": "No color frame or synced RGBD pair was captured.",
        }

    prepared = service._prepare_faces_for_recognition_result(image, depth_m)
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "accepted": False,
        "preparation": {
            "reason": prepared.reason,
            "detected_count": prepared.detected_count,
            "rejected_count": prepared.rejected_count,
            "usable_face_count": len(prepared.faces),
        },
        "artifacts": {},
    }
    source_saved = save_preview_image(
        image,
        output_dir=output_dir,
        prefix=f"{sample_id}_source",
        metadata={"sample_id": sample_id, "kind": "source"},
    )
    if source_saved:
        payload["artifacts"]["source_image_path"] = source_saved.get("image_path")
        payload["artifacts"]["source_metadata_path"] = source_saved.get("metadata_path")
    if not prepared.faces:
        payload["reason"] = prepared.reason or "no_face"
        return payload

    face, multiple_people_visible = service._select_enrollment_face(prepared.faces)
    if multiple_people_visible or face is None:
        payload["reason"] = "multiple_faces" if multiple_people_visible else "no_selected_face"
        payload["faces"] = [summarize_face(item, include_embedding=False) for item in prepared.faces]
        return payload

    quality = describe_enrollment_face_quality(service, image, face)
    preview = service._enrollment_preview_image(image, face)
    preview_saved = save_preview_image(
        preview,
        output_dir=output_dir,
        prefix=f"{sample_id}_face",
        metadata={"sample_id": sample_id, "kind": "face_preview", "quality": quality},
    )
    if preview_saved:
        payload["artifacts"]["preview_image_path"] = preview_saved.get("image_path")
        payload["artifacts"]["preview_metadata_path"] = preview_saved.get("metadata_path")
    payload["face"] = summarize_face(face, include_embedding=True)
    payload["quality"] = quality
    if not bool(quality.get("accepted")):
        payload["reason"] = str(quality.get("reason") or "quality_rejected")
        payload["message"] = str(quality.get("guidance") or "")
        return payload

    if check_remote_match:
        match, diagnostics = service._recognize_face_match_with_diagnostics(face)
        payload["existing_match"] = diagnostics
        matched_person_id = str((match or {}).get("person_id") or "").strip()
        if matched_person_id and matched_person_id != person_id:
            payload["reason"] = "matched_different_person"
            payload["message"] = f"Captured face matched {matched_person_id}, not {person_id}."
            return payload

    payload["accepted"] = True
    payload["reason"] = "accepted"
    payload["embedding"] = service._average_embeddings([face["embedding"]])
    return payload


def _collect_face_samples(
    *,
    args: argparse.Namespace,
    service: Any,
    display: Any | None,
    session_dir: Path,
    camera_resource_id: str,
    timeout_sec: float,
    person_id: str,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    samples_path = session_dir / "face_samples.jsonl"
    samples_path.write_text("", encoding="utf-8")
    photo_dir = session_dir / "biometric_photos"
    photo_dir.mkdir(parents=True, exist_ok=True)

    target = max(1, int(args.photos))
    face_intro = (
        f"I will snap {target} photos.\n"
        "Change angle a little each time: smile, tilt, crouch, or move closer."
    )
    _show(
        display,
        "Face enrollment",
        face_intro,
    )
    photo_intro = (
        f"I will snap {target} photos with a countdown before each.\n\n"
        "Change angle a little each time: smile, tilt, crouch, or move closer."
    )
    print(f"\nFace enrollment: {photo_intro}")
    if not _review_prompt(
        display,
        title="Face enrollment",
        message=photo_intro,
        accept_label="Start photos",
        reject_label="Cancel",
        assume_yes=bool(args.yes),
    ):
        raise RuntimeError("Photo capture rejected by operator.")

    for attempt in range(1, max(1, int(args.max_photo_attempts)) + 1):
        if len(accepted) >= target:
            break
        sample_number = len(accepted) + 1
        guidance = PHOTO_GUIDANCE[(sample_number - 1) % len(PHOTO_GUIDANCE)]
        sample_id = f"face_{sample_number:04d}_attempt_{attempt:04d}"
        _show(display, f"Photo {sample_number}/{target}", "Get ready.")
        print(f"[photo {sample_number}/{target}] {guidance}")
        _countdown(display, int(args.photo_countdown_sec))
        _show(display, f"Photo {sample_number}/{target}", "Hold still.")
        sample = _capture_face_sample(
            service=service,
            camera_resource_id=camera_resource_id,
            timeout_sec=timeout_sec,
            output_dir=photo_dir,
            sample_id=sample_id,
            person_id=person_id,
        )
        clean_sample = {key: value for key, value in sample.items() if key != "embedding"}
        clean_sample["captured_at_unix_s"] = round(time.time(), 3)
        append_jsonl(samples_path, json_ready(clean_sample))
        if sample.get("accepted"):
            accepted.append(sample)
            _show(display, f"Saved photo {len(accepted)}/{target}", "Good capture.")
        else:
            reason = str(sample.get("reason") or "rejected")
            message = str(sample.get("message") or "")
            _discard_sample_artifacts(sample, session_dir=session_dir)
            _show(display, "Photo not saved", message or reason)
            print(f"Rejected photo attempt {attempt}: {reason} {message}".strip())

    if len(accepted) < target:
        raise RuntimeError(f"Only collected {len(accepted)}/{target} accepted face samples.")
    return accepted[:target]


def _capture_voice_sample(
    *,
    args: argparse.Namespace,
    config: Any,
    vad: object,
    display: Any | None,
    audio_dir: Path,
    sample_id: str,
    prompt: str,
    sample_number: int,
    total: int,
    speaker_service: Any,
) -> dict[str, Any]:
    print(f"[voice] Read: {prompt}")
    title = _voice_sample_title(sample_number, total)
    sentence_detail = _voice_sentence_detail(prompt)
    _show(display, title, VOICE_COUNTDOWN_DETAIL)
    _countdown(display, int(args.voice_countdown_sec))
    _show(display, title, sentence_detail)

    def _show_recording() -> None:
        _show(display, title, sentence_detail)
        _show_subtitle(display, f"Recording audio {sample_number}/{total}")

    capture = _capture_microphone_utterance_raw(
        config,
        vad=vad,
        on_listening=(lambda: _show(display, title, sentence_detail)),
        on_recording_start=_show_recording,
        on_recording_stop=(
            (
                lambda reason: _show(
                    display,
                    f"Submitting audio {sample_number}/{total}...",
                    VOICE_SUBMITTING_DETAIL,
                )
            )
            if display is not None
            else None
        ),
    )
    if not capture.get("success"):
        return {
            "sample_id": sample_id,
            "accepted": False,
            "reason": str(capture.get("failure_reason") or "capture_failed"),
            "capture": capture,
        }

    audio_pcm16 = bytes(capture.get("agent_audio_pcm16") or b"")
    waveform = np.frombuffer(audio_pcm16, dtype=np.int16).copy()
    rejection = enrollment_rejection_reason(speaker_service.policy, audio_pcm16=waveform)
    source_audio = bytes(capture.get("source_audio_pcm16") or b"")
    source_rate = int(capture.get("source_sample_rate_hz") or 0)
    source_path = (
        write_pcm16_wav(
            audio_dir / f"{sample_id}_input_{source_rate}hz.wav",
            source_audio,
            sample_rate=source_rate,
        )
        if source_audio and source_rate > 0
        else ""
    )
    agent_path = write_pcm16_wav(audio_dir / f"{sample_id}_agent_{SAMPLE_RATE}hz.wav", audio_pcm16)
    metadata = {
        "sample_id": sample_id,
        "accepted": not bool(rejection),
        "reason": rejection or "accepted",
        "prompt": prompt,
        "capture": {
            key: value
            for key, value in capture.items()
            if key not in {"source_audio_pcm16", "agent_audio_pcm16"}
        },
        "audio": {
            "agent_stats": render_stats_payload(audio_pcm16),
            "clip_stats": asdict(clip_stats(waveform)),
        },
        "artifacts": {
            "source_wav_path": source_path,
            "agent_16k_wav_path": agent_path,
        },
    }
    metadata_path = audio_dir / f"{sample_id}.json"
    metadata["artifacts"]["metadata_path"] = str(metadata_path)
    write_json(metadata_path, json_ready(metadata))
    if rejection:
        return metadata
    embedding = speaker_service.backend.embed_query_clip(waveform, sample_rate=SAMPLE_RATE)
    metadata["embedding"] = embedding
    return metadata


def _collect_voice_samples(
    *,
    args: argparse.Namespace,
    speaker_service: Any,
    config: Any,
    vad: object,
    display: Any | None,
    session_dir: Path,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    samples_path = session_dir / "voice_samples.jsonl"
    samples_path.write_text("", encoding="utf-8")
    audio_dir = session_dir / "biometric_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    target = max(1, int(args.voice_clips))
    _show(
        display,
        "Voice enrollment",
        f"I need {target} voice recordings.\nSpeak only after the countdown.",
    )
    voice_intro = (
        f"I need {target} voice recordings.\n\n"
        "After each countdown, the sentence will appear. Start speaking then."
    )
    print(f"\nVoice enrollment: {voice_intro}")
    if not _review_prompt(
        display,
        title="Voice enrollment",
        message=voice_intro,
        accept_label="Start voice",
        reject_label="Cancel",
        assume_yes=bool(args.yes),
    ):
        raise RuntimeError("Voice capture rejected by operator.")

    for attempt in range(1, max(1, int(args.max_voice_attempts)) + 1):
        if len(accepted) >= target:
            break
        sample_number = len(accepted) + 1
        prompt = VOICE_PROMPTS[(sample_number - 1) % len(VOICE_PROMPTS)]
        sample_id = f"voice_{sample_number:04d}_attempt_{attempt:04d}"
        sample = _capture_voice_sample(
            args=args,
            config=config,
            vad=vad,
            display=display,
            audio_dir=audio_dir,
            sample_id=sample_id,
            prompt=prompt,
            sample_number=sample_number,
            total=target,
            speaker_service=speaker_service,
        )
        clean_sample = {key: value for key, value in sample.items() if key != "embedding"}
        clean_sample["captured_at_unix_s"] = round(time.time(), 3)
        append_jsonl(samples_path, json_ready(clean_sample))
        if sample.get("accepted"):
            accepted.append(sample)
            _show(display, f"Saved voice {len(accepted)}/{target}", "Good recording.")
        else:
            reason = str(sample.get("reason") or "rejected")
            _discard_sample_artifacts(sample, session_dir=session_dir)
            _show(display, "Voice not saved", reason)
            print(f"Rejected voice attempt {attempt}: {reason}")

    if len(accepted) < target:
        raise RuntimeError(f"Only collected {len(accepted)}/{target} accepted voice samples.")
    return accepted[:target]


def _embedding_consistency_summary(
    embeddings: list[Any],
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    aggregate = normalize_mean_embedding(embeddings)
    similarities: list[float] = []
    normalized: list[np.ndarray] = []
    for embedding in embeddings:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise RuntimeError("Biometric sample has a zero-norm embedding.")
        normalized.append(vector / norm)
        similarities.append(round(float(np.dot(normalized[-1], aggregate)), 4))
    pairwise_values = [
        float(np.dot(normalized[left], normalized[right]))
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    minimum = min(pairwise_values, default=1.0)
    pairwise_similarities = [round(value, 4) for value in pairwise_values]
    if minimum < float(threshold):
        raise RuntimeError(
            f"Biometric samples were inconsistent: minimum pairwise similarity {minimum:.4f} "
            f"is below {float(threshold):.4f}."
        )
    return aggregate, {
        "count": len(embeddings),
        "similarities_to_aggregate": similarities,
        "min_similarity": round(minimum, 4),
        "pairwise_similarities": pairwise_similarities,
        "threshold": float(threshold),
    }


def main() -> int:
    parser = _build_parser()
    raw_argv = sys.argv[1:]
    if "--commit" in raw_argv:
        parser.error("--commit was removed; capture locally, then run the push command.")
    args = parser.parse_args(normalize_cli_argv(raw_argv))
    configure_logging(bool(args.verbose))
    if args.list:
        return list_local_bundles(args)
    if args.push:
        return push_local_bundle(args)
    if args.cleanup:
        return cleanup_local_bundle(args)
    if not str(args.person_name or "").strip():
        parser.error("person_name is required for capture")

    from scripts.labs.face_lab_common import build_enrollment_policy, build_face_service
    if int(args.photos) < 5 or int(args.voice_clips) < 5:
        parser.error(
            "--photos and --voice-clips must each be at least 5 for identity consistency."
        )

    display = None
    face_service = None
    face_config: dict[str, Any] = {}
    speaker_service = None
    try:
        profile = load_profile(args.profile)
        site_code = str(args.site_code or profile.identity_memory.site_code or "").strip()
        if not site_code:
            parser.error("--site-code is required when the profile does not set identity_memory.site_code")

        bundle = create_bundle(
            args.output_root,
            person_name=args.person_name,
            person_id=args.person_id,
            metadata={
                "username": str(args.username or "").strip().lower(),
                "site_code": site_code,
                "profile": profile.name,
                "profile_arg": args.profile,
            },
        )
        session_dir = bundle.path
        person_id = str(args.person_id or f"local_{bundle.bundle_id}")
        metadata = {
            "site_code": site_code,
            "display_name": str(args.person_name).strip(),
            "name": str(args.person_name).strip(),
            "username": str(args.username or "").strip().lower(),
        }

        display = create_display_runtime_for_profile(
            profile,
            disabled=bool(args.no_display),
            provider_transport=args.provider_transport,
        )
        enrollment_policy = build_enrollment_policy(args)
        face_service, face_config = build_face_service(args, enrollment_policy=enrollment_policy)
        setattr(args, "session_dir", str(session_dir / "_speaker_lab_session"))
        speaker_config = build_lab_config(args)
        speaker_service = build_speaker_service(speaker_config)
        vad, vad_impl = build_vad(speaker_config.vad_threshold)
        depth_settings = face_config["depth_settings"]
        timeout_sec = (
            depth_settings.capture_timeout_sec
            if depth_settings is not None
            else float(args.capture_timeout_sec or 1.5)
        )

        bundle.manifest["metadata"]["embedding_models"] = {
            "face": "facenet_vggface2",
            "voice": str(speaker_service.policy.backend),
        }
        atomic_write_json(session_dir / BUNDLE_MANIFEST_FILENAME, bundle.manifest)
        write_session_manifest(
            session_dir=session_dir,
            filename="biometric_enrollment_manifest.json",
            payload={
                "collection_kind": "biometric_enrollment_bundle",
                "local_only": True,
                "bundle_id": bundle.bundle_id,
                "claimed_person": {
                    "name": str(args.person_name).strip(),
                    "username": str(args.username or "").strip().lower(),
                    "person_id": str(args.person_id or "").strip(),
                },
                "profile": profile.name,
                "profile_arg": args.profile,
                "site_code": site_code,
                "camera_resource_id": face_config["camera_resource_id"],
                "depth_gate": vars(depth_settings) if depth_settings is not None else None,
                "enrollment_policy": asdict(enrollment_policy),
                "audio": session_summary_payload(speaker_config, vad_impl=vad_impl),
            },
        )

        print(
            json.dumps(
                {"person": json_ready(metadata), "session_dir": str(session_dir)},
                indent=2,
                default=str,
            )
        )
        _show(display, "Biometric enrollment", f"{metadata['display_name']} | {person_id}")

        face_samples = _collect_face_samples(
            args=args,
            service=face_service,
            display=display,
            session_dir=session_dir,
            camera_resource_id=face_config["camera_resource_id"],
            timeout_sec=float(timeout_sec),
            person_id=person_id,
        )
        voice_samples = _collect_voice_samples(
            args=args,
            speaker_service=speaker_service,
            config=speaker_config,
            vad=vad,
            display=display,
            session_dir=session_dir,
        )
        face_embeddings = [sample["embedding"] for sample in face_samples]
        voice_embeddings = [sample["embedding"] for sample in voice_samples]
        _, face_consistency = _embedding_consistency_summary(
            face_embeddings,
            threshold=float(enrollment_policy.min_embedding_similarity),
        )
        _, voice_consistency = _embedding_consistency_summary(
            voice_embeddings,
            threshold=float(speaker_service.policy.query_match_threshold),
        )
        summary = {
            "bundle_id": bundle.bundle_id,
            "display_name": metadata["display_name"],
            "bundle_dir": str(session_dir),
            "face_samples": len(face_samples),
            "voice_samples": len(voice_samples),
            "face_consistency": face_consistency,
            "voice_consistency": voice_consistency,
            "local_only": True,
        }
        write_json(session_dir / "biometric_enrollment_summary.json", json_ready(summary))
        finalized = finalize_bundle(
            bundle,
            face_embeddings=face_embeddings,
            voice_embeddings=voice_embeddings,
        )
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        _show(display, "Capture saved locally", metadata["display_name"])
        print(f"Capture ready for approved push: {finalized.path}")
        return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    finally:
        try:
            if speaker_service is not None:
                speaker_service.shutdown()
            if face_service is not None:
                face_service.shutdown()
        finally:
            robot_client = face_config.get("robot_client") if "face_config" in locals() else None
            if robot_client is not None:
                robot_client.shutdown()
            if display is not None:
                display.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
