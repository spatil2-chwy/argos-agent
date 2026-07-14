#!/usr/bin/env python3
"""Live guided biometric enrollment into Tailwag.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.biometric_enrollment_lab "Jane Doe" --site-code BOS3
    poetry run python -m scripts.labs.biometric_enrollment_lab "Jane Doe" --site-code BOS3 --commit

The script is dry-run by default. Pass --commit to store one face reference and
one voice reference through Tailwag. The first accepted sample enrolls the
reference; the remaining accepted samples are offered as controlled enrollment
updates so Tailwag aggregates toward its target sample count.
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
from scripts.labs.enrollment_audio_collection import _capture_microphone_utterance_raw
from scripts.labs.enrollment_collection_common import (
    DEFAULT_COLLECTION_ROOT,
    create_display_runtime_for_profile,
    create_identity_memory_client_for_profile,
    json_ready,
    load_profile,
    resolve_collection_session,
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
    "This is a test recording for Argos voice enrollment.",
    "My voice should be recognized clearly in a normal office.",
    "I am speaking at a comfortable volume for this microphone.",
    "Argos is collecting a clean biometric voice reference.",
    "This final sentence helps confirm my speaker embedding.",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect 5 live face and voice samples and optionally commit them to Tailwag."
    )
    parser.add_argument("person_name", help="Official first and last name for the person.")
    parser.add_argument("--site-code", default="", help="Tailwag/Snowflake site code, e.g. BOS3.")
    parser.add_argument("--username", default="", help="Verified directory username, if known.")
    parser.add_argument("--person-id", default="", help="Override the Tailwag person id.")
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Allow a generated person id when directory resolution fails.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually write biometric references to Tailwag. Default is dry-run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-accept operator prompts. Use only for controlled lab runs.",
    )
    parser.add_argument("--photos", type=int, default=5, help="Accepted photo samples required.")
    parser.add_argument("--voice-clips", type=int, default=5, help="Accepted voice clips required.")
    parser.add_argument("--max-photo-attempts", type=int, default=12)
    parser.add_argument("--max-voice-attempts", type=int, default=10)
    parser.add_argument("--photo-countdown-sec", type=int, default=3)
    parser.add_argument("--voice-countdown-sec", type=int, default=3)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_COLLECTION_ROOT),
        help="Root directory for collected artifacts. Default: data_collection.",
    )
    parser.add_argument("--session-id", default="")
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


def _show(display: Any | None, message: str, subtitle: str = "") -> None:
    if display is None:
        return
    try:
        display.show_message(message)
        if subtitle:
            display.show_subtitle(subtitle, duration_ms=15000)
    except Exception:
        logger.debug("Display update failed", exc_info=True)


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
    allow_cross_match: bool = False,
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

    match, diagnostics = service._recognize_face_match_with_diagnostics(face)
    payload["existing_match"] = diagnostics
    matched_person_id = str((match or {}).get("person_id") or "").strip()
    if matched_person_id and matched_person_id != person_id and not allow_cross_match:
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

    _show(
        display,
        "Face enrollment",
        "I will snap 5 photos. Change angle a little each time: smile, tilt, crouch or move closer.",
    )
    photo_intro = (
        "I will snap 5 photos with a countdown before each.\n\n"
        "Try different angles, smile, tilt, crouch slightly, and keep your whole face visible.\n\n"
        "Once you accept, the 5 photos will happen one after another."
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

    target = max(1, int(args.photos))
    for attempt in range(1, max(1, int(args.max_photo_attempts)) + 1):
        if len(accepted) >= target:
            break
        sample_number = len(accepted) + 1
        guidance = PHOTO_GUIDANCE[(sample_number - 1) % len(PHOTO_GUIDANCE)]
        sample_id = f"face_{sample_number:04d}_attempt_{attempt:04d}"
        _show(display, f"Photo {sample_number}/{target}", guidance)
        print(f"[photo {sample_number}/{target}] {guidance}")
        _countdown(display, int(args.photo_countdown_sec))
        _show(display, "Click", f"Photo {sample_number}/{target}")
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
    speaker_service: Any,
) -> dict[str, Any]:
    print(f"[voice] Read: {prompt}")
    _show(display, "Voice enrollment", prompt)
    _countdown(display, int(args.voice_countdown_sec))
    _show(display, "Mic admission active", prompt)
    capture = _capture_microphone_utterance_raw(
        config,
        vad=vad,
        on_listening=(lambda: _show(display, "Mic admission active", prompt)),
        on_recording_start=(display.show_recording if display is not None else None),
        on_recording_stop=(
            (lambda reason: _show(display, "Saving audio...", reason))
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
    write_json(audio_dir / f"{sample_id}.json", json_ready(metadata))
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

    _show(
        display,
        "Voice enrollment",
        "Now I need 5 voice recordings. Read each sentence when it appears.",
    )
    voice_intro = (
        "Now I need 5 voice recordings.\n\n"
        "Read each sentence shown on the screen after the countdown.\n\n"
        "Once you accept, the 5 recordings will happen one after another."
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

    target = max(1, int(args.voice_clips))
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
            _show(display, "Voice not saved", reason)
            print(f"Rejected voice attempt {attempt}: {reason}")

    if len(accepted) < target:
        raise RuntimeError(f"Only collected {len(accepted)}/{target} accepted voice samples.")
    return accepted[:target]


def _operator_enrollment_evidence(person_id: str) -> dict[str, Any]:
    return {
        "owner_id": person_id,
        "owner_source": "audio_face_agree",
        "primary_face_person_id": person_id,
        "audio_speaker_id": person_id,
        "face_margin": 1.0,
        "voice_margin": 1.0,
        "recognized_count": 1,
        "unknown_count": 0,
        "enrollment_mode": "operator_controlled_live",
    }


def _commit_modality(
    *,
    identity_memory: TailwagHttpIdentityMemoryClient,
    modality: str,
    person_id: str,
    embeddings: list[Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not embeddings:
        return {"modality": modality, "saved": False, "reason": "no_embeddings"}
    first_metadata = {
        **metadata,
        "enrollment_mode": "operator_controlled_live",
        "sample_index": 1,
        "sample_count_requested": len(embeddings),
    }
    if modality == "face":
        first = identity_memory.enroll_face_reference(
            person_id=person_id,
            embedding=embeddings[0],
            metadata=first_metadata,
            consent_status="consented",
        )
    else:
        first = identity_memory.enroll_voice_reference(
            person_id=person_id,
            embedding=embeddings[0],
            metadata=first_metadata,
            consent_status="consented",
        )
    first_payload = _plain(first)
    updates: list[dict[str, Any]] = []
    if not bool(first_payload.get("saved")):
        return {"modality": modality, "enrollment": first_payload, "updates": updates}

    evidence = _operator_enrollment_evidence(person_id)
    for index, embedding in enumerate(embeddings[1:], start=2):
        update_metadata = {
            **metadata,
            "enrollment_mode": "operator_controlled_live",
            "sample_index": index,
            "sample_count_requested": len(embeddings),
        }
        if modality == "face":
            result = identity_memory.observe_face_embedding(
                person_id=person_id,
                embedding=embedding,
                evidence=evidence,
                metadata=update_metadata,
            )
        else:
            result = identity_memory.observe_voice_embedding(
                person_id=person_id,
                embedding=embedding,
                evidence=evidence,
                metadata=update_metadata,
            )
        updates.append(_plain(result))
    return {"modality": modality, "enrollment": first_payload, "updates": updates}


def _similarity_summary(service: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    embeddings = [sample.get("embedding") for sample in samples if sample.get("embedding") is not None]
    if len(embeddings) < 2:
        return {"count": len(embeddings), "pairwise_to_first": []}
    values = [
        round(float(service._embedding_similarity(embeddings[0], embedding)), 4)
        for embedding in embeddings
    ]
    return {
        "count": len(embeddings),
        "pairwise_to_first": values,
        "min_to_first": min(values),
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    from scripts.labs.face_lab_common import build_enrollment_policy, build_face_service

    configure_logging(bool(args.verbose))
    identity_memory: TailwagHttpIdentityMemoryClient | None = None
    display = None
    face_service = None
    face_config: dict[str, Any] = {}
    speaker_service = None
    try:
        profile = load_profile(args.profile)
        site_code = str(args.site_code or profile.identity_memory.site_code or "").strip()
        if not site_code:
            parser.error("--site-code is required when the profile does not set identity_memory.site_code")

        session = resolve_collection_session(
            output_root=args.output_root,
            person_name=args.person_name,
            person_id=args.person_id,
            session_id=args.session_id,
        )
        session_dir = Path(session["session_dir"])
        session_dir.mkdir(parents=True, exist_ok=True)

        identity_memory = create_identity_memory_client_for_profile(
            profile,
            site_code=site_code,
        )
        person = _identity_from_args(args, identity_memory=identity_memory, site_code=site_code)
        person_id = str(person.get("person_id") or "").strip()
        if not person_id:
            raise RuntimeError("Resolved person has no person_id.")
        metadata = {
            **dict(person),
            "site_code": site_code,
            "display_name": str(
                person.get("display_name") or person.get("official_name") or args.person_name
            ),
            "name": str(
                person.get("display_name") or person.get("official_name") or args.person_name
            ),
        }

        display = create_display_runtime_for_profile(
            profile,
            disabled=bool(args.no_display),
            provider_transport=args.provider_transport,
        )
        enrollment_policy = build_enrollment_policy(args)
        face_service, face_config = build_face_service(args, enrollment_policy=enrollment_policy)
        face_service.identity_memory_client = identity_memory
        setattr(args, "session_dir", str(session_dir / "_speaker_lab_session"))
        speaker_config = build_lab_config(args)
        speaker_service = build_speaker_service(speaker_config)
        speaker_service.identity_memory_client = identity_memory
        vad, vad_impl = build_vad(speaker_config.vad_threshold)
        depth_settings = face_config["depth_settings"]
        timeout_sec = (
            depth_settings.capture_timeout_sec
            if depth_settings is not None
            else float(args.capture_timeout_sec or 1.5)
        )

        write_session_manifest(
            session_dir=session_dir,
            filename="biometric_enrollment_manifest.json",
            payload={
                "collection_kind": "biometric_enrollment",
                "dry_run": not bool(args.commit),
                "person": {key: value for key, value in metadata.items() if key != "resolution"},
                "identity_resolution": metadata.get("resolution"),
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
        summary = {
            "person_id": person_id,
            "display_name": metadata["display_name"],
            "session_dir": str(session_dir),
            "face_samples": len(face_samples),
            "voice_samples": len(voice_samples),
            "face_similarity": _similarity_summary(face_service, face_samples),
            "dry_run": not bool(args.commit),
        }
        write_json(session_dir / "biometric_enrollment_summary.json", json_ready(summary))
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))

        if not args.commit:
            _show(display, "Dry run complete", "No Tailwag writes were made.")
            print("Dry run complete. Re-run with --commit to write references to Tailwag.")
            return 0
        save_message = (
            "The face and voice samples are complete.\n\n"
            "Accept to save one face reference and one voice reference to Tailwag."
        )
        if not _review_prompt(
            display,
            title="Save enrollment",
            message=save_message,
            accept_label="Save",
            reject_label="Do not save",
            assume_yes=bool(args.yes),
        ):
            _show(display, "Commit skipped", "No Tailwag writes were made.")
            return 0

        face_commit = _commit_modality(
            identity_memory=identity_memory,
            modality="face",
            person_id=person_id,
            embeddings=[sample["embedding"] for sample in face_samples],
            metadata=metadata,
        )
        voice_commit = _commit_modality(
            identity_memory=identity_memory,
            modality="voice",
            person_id=person_id,
            embeddings=[sample["embedding"] for sample in voice_samples],
            metadata=metadata,
        )
        commit_payload = {
            "person_id": person_id,
            "face": face_commit,
            "voice": voice_commit,
            "note": (
                "This lab creates new active biometric references. Archive old bad references "
                "with Tailwag tooling if duplicate same-person references hurt margins."
            ),
        }
        write_json(session_dir / "biometric_enrollment_commit.json", json_ready(commit_payload))
        print(json.dumps(json_ready(commit_payload), indent=2, sort_keys=True))
        _show(display, "Enrollment saved", f"{metadata['display_name']} | {person_id}")
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
            if identity_memory is not None:
                identity_memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
