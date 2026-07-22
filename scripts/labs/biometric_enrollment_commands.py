#!/usr/bin/env python3
"""Operator commands for local biometric enrollment bundles."""

from __future__ import annotations

import argparse
import getpass
import json
import time
from typing import Any, Callable

from argos_src.identity_memory import TailwagHttpIdentityMemoryClient
from scripts.labs.biometric_enrollment_bundle import (
    BiometricEnrollmentBundle,
    bind_bundle_person,
    cleanup_bundle,
    discover_bundles,
    discover_completed_bundles,
    discover_invalid_bundle_paths,
    is_bundle_ready_for_upload,
    is_cleanup_eligible,
    load_aggregate_embeddings,
    load_bundle,
    update_upload_state,
    upload_complete,
)
from scripts.labs.enrollment_collection_common import (
    create_identity_memory_client_for_profile,
    load_profile,
)


def normalize_cli_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if argv[0] == "capture":
        return argv[1:]
    if argv[0] in {"list", "push", "cleanup"}:
        return [f"--{argv[0]}", *argv[1:]]
    return argv


def bundle_label(bundle: BiometricEnrollmentBundle) -> str:
    person = dict(bundle.manifest.get("person") or {})
    name = str(person.get("name") or "Unknown")
    created_at = str(bundle.manifest.get("created_at") or "unknown time")
    states = ", ".join(
        f"{modality}={state}" for modality, state in bundle.modality_states.items()
    )
    capture_state = str(bundle.manifest.get("status") or "unknown")
    return (
        f"{name} | {created_at} | capture={capture_state} | {states} | "
        f"id={bundle.bundle_id}"
    )


def select_bundle(
    bundles: list[BiometricEnrollmentBundle],
    *,
    action: str,
    input_fn: Callable[[str], str] | None = None,
) -> BiometricEnrollmentBundle | None:
    if not bundles:
        print(f"No local biometric bundles are eligible to {action}.")
        return None
    reader = input_fn or input
    for index, bundle in enumerate(bundles, start=1):
        print(f"{index}. {bundle_label(bundle)}")
    answer = str(
        reader(f"Choose a person to {action} [1-{len(bundles)}] or Enter to cancel: ")
    ).strip()
    if not answer:
        return None
    try:
        selected = int(answer)
    except ValueError as exc:
        raise RuntimeError("Selection must be a number.") from exc
    if selected < 1 or selected > len(bundles):
        raise RuntimeError("Selection is outside the displayed range.")
    return bundles[selected - 1]


def _bundle_identity_args(bundle: BiometricEnrollmentBundle) -> argparse.Namespace:
    person = dict(bundle.manifest.get("person") or {})
    metadata = dict(bundle.manifest.get("metadata") or {})
    return argparse.Namespace(
        person_name=str(person.get("name") or "").strip(),
        username=str(metadata.get("username") or "").strip(),
        person_id="",
        allow_unresolved=False,
    )


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _validate_enrollable_profile(payload: dict[str, Any], *, person_id: str) -> None:
    metadata = dict(payload.get("metadata") or {})
    status = str(payload.get("status") or metadata.get("status") or "")
    consent_status = str(
        payload.get("consent_status") or metadata.get("consent_status") or ""
    )
    if status.strip().casefold() != "active":
        raise RuntimeError(
            f"Tailwag person {person_id} is not active and cannot receive biometrics."
        )
    if consent_status.strip().casefold() not in {"", "consented"}:
        raise RuntimeError(
            f"Tailwag person {person_id} has a non-enrollable consent status."
        )


def _resolve_bundle_identity(
    identity_memory: TailwagHttpIdentityMemoryClient,
    *,
    bundle: BiometricEnrollmentBundle,
    site_code: str,
) -> dict[str, Any]:
    person = dict(bundle.manifest.get("person") or {})
    claimed_person_id = str(person.get("person_id") or "").strip()
    claimed_name = str(person.get("name") or "").strip()
    if claimed_person_id:
        profile = identity_memory.person_profile(claimed_person_id)
        if profile is None:
            raise RuntimeError(
                f"Tailwag has no person profile for claimed id {claimed_person_id}."
            )
        payload = _plain(profile)
        _validate_enrollable_profile(payload, person_id=claimed_person_id)
        canonical_name = str(
            payload.get("display_name") or payload.get("official_name") or ""
        ).strip()
        if (
            claimed_name
            and canonical_name
            and claimed_name.casefold() != canonical_name.casefold()
        ):
            raise RuntimeError(
                f"Claimed name {claimed_name!r} does not match Tailwag profile "
                f"{canonical_name!r} for {claimed_person_id}."
            )
        return {**payload, "site_code": site_code}

    from scripts.labs.biometric_enrollment_lab import _identity_from_args

    resolved = _identity_from_args(
        _bundle_identity_args(bundle),
        identity_memory=identity_memory,
        site_code=site_code,
    )

    resolved_person_id = str(resolved.get("person_id") or "").strip()
    if not resolved_person_id:
        raise RuntimeError("Directory resolution did not return a canonical person id.")
    profile = identity_memory.person_profile(resolved_person_id)
    if profile is None:
        return {**resolved, "site_code": site_code}
    payload = _plain(profile)
    _validate_enrollable_profile(payload, person_id=resolved_person_id)
    canonical_name = str(
        payload.get("display_name") or payload.get("official_name") or ""
    ).strip()
    if (
        claimed_name
        and canonical_name
        and claimed_name.casefold() != canonical_name.casefold()
    ):
        raise RuntimeError(
            f"Captured name {claimed_name!r} does not match Tailwag profile "
            f"{canonical_name!r} for {resolved_person_id}."
        )
    return {**resolved, **payload, "site_code": site_code}


def _ensure_no_cross_person_match(
    identity_memory: TailwagHttpIdentityMemoryClient,
    *,
    modality: str,
    embedding: Any,
    person_id: str,
) -> None:
    result = (
        identity_memory.search_face(
            embedding=embedding,
            limit=2,
            global_scope=True,
            strict_response=True,
        )
        if modality == "face"
        else identity_memory.search_voice(
            embedding=embedding,
            limit=2,
            global_scope=True,
            strict_response=True,
        )
    )
    payload = _plain(result)
    reason = payload.get("reason")
    recognized = payload.get("recognized")
    raw_candidates = payload.get("candidates")
    threshold = payload.get("threshold")
    if reason == "tailwag_unavailable":
        raise RuntimeError(f"Tailwag {modality} conflict search was unavailable.")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(recognized, bool)
        or not isinstance(raw_candidates, (list, tuple))
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
    ):
        raise RuntimeError(f"Tailwag {modality} conflict search was malformed.")
    if recognized and not raw_candidates:
        raise RuntimeError(
            f"Tailwag {modality} conflict search recognized an empty candidate set."
        )
    for candidate in raw_candidates:
        item = dict(candidate)
        matched_person_id = str(item.get("person_id") or "").strip()
        score = item.get("score")
        if (
            not matched_person_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise RuntimeError(f"Tailwag {modality} conflict candidate was malformed.")
        if matched_person_id != person_id and float(score) >= float(threshold):
            raise RuntimeError(
                f"The {modality} aggregate strongly matches {matched_person_id}, "
                f"not {person_id}."
            )


def list_local_bundles(args: argparse.Namespace) -> int:
    bundles = discover_bundles(args.output_root)
    invalid_paths = discover_invalid_bundle_paths(args.output_root)
    if not bundles and not invalid_paths:
        print("No local biometric bundles found.")
        return 0
    for bundle in bundles:
        print(bundle_label(bundle))
    for path in invalid_paths:
        print(f"INVALID local biometric bundle; administrator review required: {path}")
    return 0


def push_local_bundle(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    bundles = [
        bundle
        for bundle in discover_completed_bundles(args.output_root)
        if is_bundle_ready_for_upload(bundle)
    ]
    bundle = select_bundle(bundles, action="push", input_fn=input_fn)
    if bundle is None:
        return 0
    verified = load_bundle(bundle.path, output_root=args.output_root, verify_artifacts=True)
    bundle_metadata = dict(verified.manifest.get("metadata") or {})
    sample_counts = dict(bundle_metadata.get("sample_counts") or {})
    if any(int(sample_counts.get(modality) or 0) < 5 for modality in ("face", "voice")):
        raise RuntimeError(
            "Biometric bundles require at least five accepted face and voice samples."
        )
    profile = load_profile(str(bundle_metadata.get("profile_arg") or args.profile))
    site_code = str(
        args.site_code
        or bundle_metadata.get("site_code")
        or profile.identity_memory.site_code
        or ""
    ).strip()
    if not site_code:
        raise RuntimeError("The bundle and selected profile do not provide a site code.")

    identity_memory = create_identity_memory_client_for_profile(profile, site_code=site_code)
    try:
        person = _resolve_bundle_identity(
            identity_memory=identity_memory,
            bundle=verified,
            site_code=site_code,
        )
        person_id = str(person.get("person_id") or "").strip()
        canonical_name = str(
            person.get("display_name") or person.get("official_name") or person_id
        ).strip()
        if not person_id or not canonical_name:
            raise RuntimeError("Tailwag identity resolution did not return a usable person.")
        bound_person_id = str(verified.upload_state.get("person_id") or "").strip()
        if bound_person_id and bound_person_id != person_id:
            raise RuntimeError(
                f"Bundle is bound to {bound_person_id}; refusing resolved person {person_id}."
            )

        remote_exists: dict[str, bool] = {}
        for modality in ("face", "voice"):
            if verified.modality_states[modality] in {"uploaded", "skipped"}:
                continue
            remote_exists[modality] = identity_memory.biometric_reference_exists(
                modality=modality,
                person_id=person_id,
            )
        missing = [name for name, exists in remote_exists.items() if not exists]
        reader = input_fn or input
        if missing:
            print(f"Verified person: {canonical_name} ({person_id})")
            print(f"Missing references: {', '.join(missing)}")
            approval = str(
                reader(f"Confirm subject consent; type {canonical_name!r} to approve upload: ")
            ).strip()
            if approval.casefold() != canonical_name.casefold():
                print("Upload cancelled; approval text did not match.")
                return 0

        approved_at = round(time.time(), 3)
        verified = bind_bundle_person(verified, person_id)
        approved_by = getpass.getuser()
        embeddings = load_aggregate_embeddings(verified)
        for modality in missing:
            try:
                _ensure_no_cross_person_match(
                    identity_memory,
                    modality=modality,
                    embedding=embeddings[modality],
                    person_id=person_id,
                )
            except Exception as exc:
                update_upload_state(
                    verified, modality, "failed", error=str(exc),
                    details={"person_id": person_id},
                )
                raise
        for modality in ("face", "voice"):
            if verified.modality_states[modality] in {"uploaded", "skipped"}:
                continue
            if remote_exists[modality]:
                verified = update_upload_state(
                    verified,
                    modality,
                    "skipped",
                    details={"reason": "already_exists", "person_id": person_id},
                )
                print(f"Skipped {modality}: Tailwag already has an active reference.")
                continue
            try:
                enrollment_metadata = {
                    "source": "argos_biometric_enrollment_lab",
                    "bundle_id": verified.bundle_id,
                    "display_name": canonical_name,
                    "official_name": str(person.get("official_name") or canonical_name),
                    "username": str(person.get("username") or ""),
                    "site_code": site_code,
                    "operator_approved_by": approved_by,
                    "operator_approved_at_unix_s": approved_at,
                    "model": dict(
                        bundle_metadata.get("embedding_models") or {}
                    ).get(modality),
                }
                result = (
                    identity_memory.enroll_face_reference(
                        person_id=person_id,
                        embedding=embeddings[modality],
                        metadata=enrollment_metadata,
                        consent_status="consented",
                    )
                    if modality == "face"
                    else identity_memory.enroll_voice_reference(
                        person_id=person_id,
                        embedding=embeddings[modality],
                        metadata=enrollment_metadata,
                        consent_status="consented",
                    )
                )
                result_payload = _plain(result)
                if not bool(result_payload.get("saved")):
                    raise RuntimeError(
                        f"Tailwag rejected {modality} enrollment: "
                        f"{result_payload.get('reason') or 'unknown reason'}"
                    )
                returned_person_id = str(
                    result_payload.get("person_id") or ""
                ).strip()
                if returned_person_id != person_id:
                    raise RuntimeError(
                        f"Tailwag returned enrollment for {returned_person_id or 'no person'}."
                    )
                verified = update_upload_state(
                    verified,
                    modality,
                    "uploaded",
                    details={
                        "person_id": person_id,
                        "approved_by": approved_by,
                        "approved_at_unix_s": approved_at,
                        "result": result_payload,
                    },
                )
                print(f"Uploaded {modality} reference for {canonical_name}.")
            except Exception as exc:
                update_upload_state(
                    verified,
                    modality,
                    "failed",
                    error=str(exc),
                    details={"person_id": person_id},
                )
                raise
        if upload_complete(verified):
            print(f"Biometric push complete for {canonical_name}.")
        return 0
    finally:
        identity_memory.close()


def cleanup_local_bundle(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    bundles = [
        bundle
        for bundle in discover_completed_bundles(args.output_root)
        if is_cleanup_eligible(bundle)
    ]
    bundle = select_bundle(bundles, action="delete", input_fn=input_fn)
    if bundle is None:
        return 0
    person_name = str((bundle.manifest.get("person") or {}).get("name") or "")
    reader = input_fn or input
    approval = str(
        reader(f"Type 'DELETE {person_name}' to remove all local biometrics: ")
    ).strip()
    if approval != f"DELETE {person_name}":
        print("Cleanup cancelled.")
        return 0
    cleanup_bundle(args.output_root, bundle.path)
    print(f"Deleted local biometric bundle for {person_name}.")
    return 0
