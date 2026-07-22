#!/usr/bin/env python3
"""Local, offline storage for completed biometric enrollment captures.

Bundles deliberately contain raw, unencrypted lab artifacts.  This module does
not capture from hardware or upload over the network; it only manages files
below a caller-provided collection root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4

import numpy as np


SCHEMA_VERSION = 1
BUNDLE_STORE_DIRECTORY = ".biometric_enrollment_bundles"
BUNDLE_MANIFEST_FILENAME = "bundle.json"
UPLOAD_STATE_FILENAME = "upload_state.json"
EMBEDDINGS_FILENAME = "biometric_embeddings.npz"

MODALITIES = ("face", "voice")
UPLOAD_STATES = frozenset({"pending", "uploaded", "skipped", "failed"})
SUCCESSFUL_UPLOAD_STATES = frozenset({"uploaded", "skipped"})
_CONTROL_FILENAMES = frozenset({BUNDLE_MANIFEST_FILENAME, UPLOAD_STATE_FILENAME})


class BundleError(ValueError):
    """Raised when a biometric bundle is invalid or unsafe to operate on."""


@dataclass(frozen=True)
class BiometricEnrollmentBundle:
    """A validated view of one local enrollment bundle."""

    output_root: Path
    path: Path
    manifest: dict[str, Any]
    upload_state: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def completed(self) -> bool:
        return self.manifest.get("status") == "complete"

    @property
    def modality_states(self) -> dict[str, str]:
        return {
            modality: str(self.upload_state["modalities"][modality]["state"])
            for modality in MODALITIES
        }

    @property
    def ready_for_upload(self) -> bool:
        return self.completed and any(
            state in {"pending", "failed"} for state in self.modality_states.values()
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a JSON file with a fully flushed temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Could not read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"Expected a JSON object in {path}")
    return payload


def _validate_uuid(value: Any) -> str:
    rendered = str(value or "")
    try:
        parsed = UUID(rendered)
    except ValueError as exc:
        raise BundleError(f"Invalid bundle UUID: {rendered!r}") from exc
    if parsed.version != 4 or str(parsed) != rendered:
        raise BundleError(f"Bundle id must be a canonical UUID4: {rendered!r}")
    return rendered


def _bundle_store(output_root: str | Path) -> Path:
    store = Path(output_root).expanduser().resolve() / BUNDLE_STORE_DIRECTORY
    if store.is_symlink():
        raise BundleError(
            f"Biometric bundle store may not be a symbolic link: {store}"
        )
    return store


def _validate_schema(payload: Mapping[str, Any], *, source: Path) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(
            f"Unsupported schema_version in {source}: {payload.get('schema_version')!r}"
        )


def create_bundle(
    output_root: str | Path,
    *,
    person_name: str,
    person_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> BiometricEnrollmentBundle:
    """Create an empty schema-v1 bundle with an internally generated UUID4."""

    rendered_name = str(person_name or "").strip()
    if not rendered_name:
        raise BundleError("person_name is required")
    bundle_id = str(uuid4())
    root = Path(output_root).expanduser().resolve()
    bundle_path = _bundle_store(root) / bundle_id
    bundle_path.mkdir(parents=True, mode=0o700)
    created_at = _utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "status": "collecting",
        "created_at": created_at,
        "completed_at": None,
        "person": {
            "name": rendered_name,
            "person_id": str(person_id or "").strip(),
        },
        "metadata": _json_safe(dict(metadata or {})),
        "artifacts": [],
    }
    upload_state = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "person_id": None,
        "updated_at": created_at,
        "modalities": {
            modality: {
                "state": "pending",
                "error": None,
                "details": None,
                "updated_at": created_at,
            }
            for modality in MODALITIES
        },
    }
    try:
        atomic_write_json(bundle_path / BUNDLE_MANIFEST_FILENAME, manifest)
        atomic_write_json(bundle_path / UPLOAD_STATE_FILENAME, upload_state)
    except Exception:
        shutil.rmtree(bundle_path, ignore_errors=True)
        raise
    return BiometricEnrollmentBundle(root, bundle_path, manifest, upload_state)


def _as_embedding(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise BundleError(f"{label} must be a non-empty one-dimensional embedding")
    if not np.all(np.isfinite(vector)):
        raise BundleError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise BundleError(f"{label} has zero norm")
    return vector


def normalize_mean_embedding(embeddings: Iterable[Any]) -> np.ndarray:
    """L2-normalize every sample, average all samples, then normalize the mean."""

    vectors = [_as_embedding(item, label=f"embedding[{index}]") for index, item in enumerate(embeddings)]
    if not vectors:
        raise BundleError("At least one embedding is required")
    expected_shape = vectors[0].shape
    if any(vector.shape != expected_shape for vector in vectors[1:]):
        raise BundleError("All embeddings must have the same shape")
    normalized = np.stack([vector / np.linalg.norm(vector) for vector in vectors])
    mean = normalized.mean(axis=0, dtype=np.float32)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 0.0:
        raise BundleError("Normalized embeddings cancel to a zero mean")
    return np.asarray(mean / mean_norm, dtype=np.float32)


def _resolve_bundle_path(bundle: str | Path | BiometricEnrollmentBundle) -> Path:
    return bundle.path if isinstance(bundle, BiometricEnrollmentBundle) else Path(bundle)


def save_aggregate_embeddings(
    bundle: str | Path | BiometricEnrollmentBundle,
    *,
    face_embeddings: Iterable[Any],
    voice_embeddings: Iterable[Any],
) -> Path:
    """Aggregate and atomically save face and voice vectors as float32 NPZ data."""

    bundle_path = _resolve_bundle_path(bundle)
    face = normalize_mean_embedding(face_embeddings)
    voice = normalize_mean_embedding(voice_embeddings)
    target = bundle_path / EMBEDDINGS_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=bundle_path,
            prefix=f".{EMBEDDINGS_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(handle, face=face, voice=voice)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def load_aggregate_embeddings(
    bundle: str | Path | BiometricEnrollmentBundle,
) -> dict[str, np.ndarray]:
    """Load and validate the aggregate face and voice vectors from a bundle."""

    target = _resolve_bundle_path(bundle) / EMBEDDINGS_FILENAME
    try:
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != set(MODALITIES):
                raise BundleError(f"{target} must contain exactly face and voice vectors")
            vectors = {
                modality: _as_embedding(archive[modality], label=modality).astype(
                    np.float32, copy=False
                )
                for modality in MODALITIES
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"Could not load biometric embeddings from {target}") from exc
    return vectors


def _artifact_paths(bundle_path: Path) -> list[Path]:
    paths: list[Path] = []
    for path in bundle_path.rglob("*"):
        if path.is_symlink():
            raise BundleError(f"Bundle artifacts may not be symbolic links: {path}")
        if not path.is_file() or path.name in _CONTROL_FILENAMES or path.name.endswith(".tmp"):
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(bundle_path).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_inventory(
    bundle: str | Path | BiometricEnrollmentBundle,
) -> list[dict[str, Any]]:
    """Build a stable SHA-256 inventory of every non-control artifact."""

    bundle_path = _resolve_bundle_path(bundle)
    return [
        {
            "path": path.relative_to(bundle_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _artifact_paths(bundle_path)
    ]


def verify_artifact_inventory(
    bundle: str | Path | BiometricEnrollmentBundle,
    manifest: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the exact current artifact set matches the saved inventory."""

    bundle_path = _resolve_bundle_path(bundle)
    try:
        expected_manifest = dict(manifest or _read_json(bundle_path / BUNDLE_MANIFEST_FILENAME))
        expected = expected_manifest.get("artifacts")
        if not isinstance(expected, list):
            return False
        return build_artifact_inventory(bundle_path) == expected
    except (BundleError, OSError):
        return False


def finalize_bundle(
    bundle: str | Path | BiometricEnrollmentBundle,
    *,
    face_embeddings: Iterable[Any] | None = None,
    voice_embeddings: Iterable[Any] | None = None,
) -> BiometricEnrollmentBundle:
    """Save aggregates if supplied, inventory artifacts, and mark capture complete."""

    bundle_path = _resolve_bundle_path(bundle)
    manifest = _read_json(bundle_path / BUNDLE_MANIFEST_FILENAME)
    _validate_schema(manifest, source=bundle_path / BUNDLE_MANIFEST_FILENAME)
    bundle_id = _validate_uuid(manifest.get("bundle_id"))
    if bundle_path.name != bundle_id:
        raise BundleError("Bundle directory does not match its bundle_id")
    if manifest.get("status") == "complete":
        raise BundleError("A completed bundle is immutable and cannot be finalized again")
    if (face_embeddings is None) != (voice_embeddings is None):
        raise BundleError("face_embeddings and voice_embeddings must be supplied together")
    sample_counts: dict[str, int] | None = None
    if face_embeddings is not None and voice_embeddings is not None:
        face_values = list(face_embeddings)
        voice_values = list(voice_embeddings)
        sample_counts = {"face": len(face_values), "voice": len(voice_values)}
        if min(sample_counts.values()) < 5:
            raise BundleError(
                "Completed bundles require at least five face and voice samples"
            )
        save_aggregate_embeddings(
            bundle_path,
            face_embeddings=face_values,
            voice_embeddings=voice_values,
        )
    load_aggregate_embeddings(bundle_path)
    completed_metadata = dict(manifest.get("metadata") or {})
    if sample_counts is not None:
        completed_metadata["sample_counts"] = sample_counts
    completed = {
        **manifest,
        "metadata": completed_metadata,
        "status": "complete",
        "completed_at": _utc_now(),
        "artifacts": build_artifact_inventory(bundle_path),
    }
    atomic_write_json(bundle_path / BUNDLE_MANIFEST_FILENAME, completed)
    return load_bundle(bundle_path, output_root=bundle_path.parent.parent)


def _validate_upload_state(payload: Mapping[str, Any], *, source: Path, bundle_id: str) -> None:
    _validate_schema(payload, source=source)
    if payload.get("bundle_id") != bundle_id:
        raise BundleError(f"Upload state bundle_id does not match {bundle_id}")
    modalities = payload.get("modalities")
    bound_person_id = payload.get("person_id")
    if bound_person_id is not None and not str(bound_person_id).strip():
        raise BundleError("Upload-state person_id must be a non-empty string or null")

    if not isinstance(modalities, dict) or set(modalities) != set(MODALITIES):
        raise BundleError("Upload state must contain exactly face and voice modalities")
    for modality in MODALITIES:
        entry = modalities[modality]
        if not isinstance(entry, dict) or entry.get("state") not in UPLOAD_STATES:
            raise BundleError(f"Invalid upload state for {modality}")
        if entry.get("details") is not None and not isinstance(entry.get("details"), dict):
            raise BundleError(f"Upload details for {modality} must be an object or null")


def load_bundle(
    bundle_path: str | Path,
    *,
    output_root: str | Path | None = None,
    verify_artifacts: bool = True,
) -> BiometricEnrollmentBundle:
    """Load one bundle and validate schema, ids, upload state, and completed artifacts."""

    path = Path(bundle_path).expanduser().resolve()
    manifest_path = path / BUNDLE_MANIFEST_FILENAME
    state_path = path / UPLOAD_STATE_FILENAME
    manifest = _read_json(manifest_path)
    _validate_schema(manifest, source=manifest_path)
    bundle_id = _validate_uuid(manifest.get("bundle_id"))
    if path.name != bundle_id:
        raise BundleError("Bundle directory does not match its bundle_id")
    upload_state = _read_json(state_path)
    _validate_upload_state(upload_state, source=state_path, bundle_id=bundle_id)
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else path.parent.parent.resolve()
    )
    if path.parent != _bundle_store(root):
        raise BundleError(f"Bundle is not rooted under configured output root: {root}")
    status = manifest.get("status")
    if status not in {"collecting", "complete"}:
        raise BundleError(f"Invalid bundle status: {status!r}")
    if status == "complete":
        sample_counts = dict((manifest.get("metadata") or {}).get("sample_counts") or {})
        for modality in MODALITIES:
            count = sample_counts.get(modality)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 5
            ):
                raise BundleError(
                    f"Completed bundle has an invalid {modality} sample count"
                )
        load_aggregate_embeddings(path)
        if verify_artifacts and not verify_artifact_inventory(path, manifest):
            raise BundleError("Bundle artifact inventory verification failed")
    return BiometricEnrollmentBundle(root, path, manifest, upload_state)


def discover_invalid_bundle_paths(output_root: str | Path) -> list[Path]:
    """List bundle-store entries that normal verified discovery cannot load."""

    root = Path(output_root).expanduser().resolve()
    store = _bundle_store(root)
    if not store.exists():
        return []
    valid_paths = {
        bundle.path
        for bundle in discover_bundles(root, verify_artifacts=True)
    }
    return sorted(
        (
            path
            for path in store.iterdir()
            if (path.is_dir() or path.is_symlink()) and path.resolve() not in valid_paths
        ),
        key=lambda path: path.name,
    )


def discover_bundles(
    output_root: str | Path,
    *,
    verify_artifacts: bool = True,
) -> list[BiometricEnrollmentBundle]:
    """Discover every valid bundle, ignoring corrupt or unsupported entries."""

    root = Path(output_root).expanduser().resolve()
    store = _bundle_store(root)
    if not store.exists():
        return []
    discovered: list[BiometricEnrollmentBundle] = []
    for manifest_path in sorted(store.glob(f"*/{BUNDLE_MANIFEST_FILENAME}")):
        try:
            bundle = load_bundle(
                manifest_path.parent,
                output_root=root,
                verify_artifacts=verify_artifacts,
            )
        except BundleError:
            continue
        discovered.append(bundle)
    return sorted(discovered, key=lambda item: (item.manifest.get("created_at", ""), item.bundle_id))


def discover_completed_bundles(
    output_root: str | Path,
    *,
    verify_artifacts: bool = True,
) -> list[BiometricEnrollmentBundle]:
    """Discover valid bundles whose local capture and artifact inventory are complete."""

    return [
        bundle
        for bundle in discover_bundles(output_root, verify_artifacts=verify_artifacts)
        if bundle.completed
    ]


def bind_bundle_person(
    bundle: str | Path | BiometricEnrollmentBundle,
    person_id: str,
) -> BiometricEnrollmentBundle:
    """Atomically bind a bundle's retries to one canonical Tailwag person."""

    rendered_person_id = str(person_id or "").strip()
    if not rendered_person_id:
        raise BundleError("A non-empty person_id is required to bind a bundle")
    bundle_path = _resolve_bundle_path(bundle).expanduser().resolve()
    loaded = load_bundle(bundle_path, verify_artifacts=False)
    current = str(loaded.upload_state.get("person_id") or "").strip()
    if current and current != rendered_person_id:
        raise BundleError(
            f"Bundle is already bound to {current}; refusing person {rendered_person_id}"
        )
    if current == rendered_person_id:
        return load_bundle(
            bundle_path,
            output_root=loaded.output_root,
            verify_artifacts=loaded.completed,
        )
    updated = json.loads(json.dumps(loaded.upload_state))
    updated["person_id"] = rendered_person_id
    updated["updated_at"] = _utc_now()
    atomic_write_json(bundle_path / UPLOAD_STATE_FILENAME, updated)
    return load_bundle(
        bundle_path, output_root=loaded.output_root, verify_artifacts=loaded.completed
    )
def update_upload_state(
    bundle: str | Path | BiometricEnrollmentBundle,
    modality: str,
    state: str,
    *,
    error: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> BiometricEnrollmentBundle:
    """Atomically transition one modality.s local upload state."""

    rendered_modality = str(modality or "").strip().lower()
    rendered_state = str(state or "").strip().lower()
    if rendered_modality not in MODALITIES:
        raise BundleError(f"Unknown modality: {modality!r}")
    if rendered_state not in UPLOAD_STATES:
        raise BundleError(f"Unknown upload state: {state!r}")
    if details is not None and not isinstance(details, Mapping):
        raise BundleError("Upload details must be a mapping or null")
    bundle_path = _resolve_bundle_path(bundle).expanduser().resolve()
    loaded = load_bundle(bundle_path, verify_artifacts=False)
    current_state = loaded.modality_states[rendered_modality]
    if current_state in SUCCESSFUL_UPLOAD_STATES and rendered_state != current_state:
        raise BundleError(
            f"Refusing to transition terminal {rendered_modality} state "
            f"from {current_state} to {rendered_state}"
        )
    now = _utc_now()
    updated = json.loads(json.dumps(loaded.upload_state))
    updated["updated_at"] = now
    updated["modalities"][rendered_modality] = {
        "state": rendered_state,
        "error": str(error) if error is not None and rendered_state == "failed" else None,
        "details": _json_safe(dict(details)) if details is not None else None,
        "updated_at": now,
    }
    atomic_write_json(bundle_path / UPLOAD_STATE_FILENAME, updated)
    return load_bundle(
        bundle_path,
        output_root=loaded.output_root,
        verify_artifacts=loaded.completed,
    )


def upload_complete(bundle: str | Path | BiometricEnrollmentBundle) -> bool:
    """Return whether face and voice are each uploaded or intentionally skipped."""

    try:
        loaded = (
            bundle
            if isinstance(bundle, BiometricEnrollmentBundle)
            else load_bundle(bundle, verify_artifacts=False)
        )
        modalities = loaded.upload_state["modalities"]
        return all(modalities[name]["state"] in SUCCESSFUL_UPLOAD_STATES for name in MODALITIES)
    except (BundleError, KeyError, TypeError):
        return False


def is_bundle_ready_for_upload(bundle: str | Path | BiometricEnrollmentBundle) -> bool:
    """Return whether an intact completed capture still has work to upload or retry."""

    try:
        loaded = load_bundle(
            _resolve_bundle_path(bundle),
            output_root=(bundle.output_root if isinstance(bundle, BiometricEnrollmentBundle) else None),
            verify_artifacts=True,
        )
    except BundleError:
        return False
    return loaded.ready_for_upload


def is_cleanup_eligible(bundle: str | Path | BiometricEnrollmentBundle) -> bool:
    """Return whether a finalized, intact bundle has no pending/failed uploads."""

    try:
        loaded = load_bundle(
            _resolve_bundle_path(bundle),
            output_root=(bundle.output_root if isinstance(bundle, BiometricEnrollmentBundle) else None),
            verify_artifacts=True,
        )
    except BundleError:
        return False
    return loaded.completed and upload_complete(loaded)


def cleanup_bundle(output_root: str | Path, bundle_path: str | Path) -> None:
    """Recursively remove only an intact, upload-complete bundle under ``output_root``."""

    root = Path(output_root).expanduser().resolve()
    path = Path(bundle_path).expanduser().resolve()
    if path.parent != _bundle_store(root):
        raise BundleError(f"Refusing cleanup outside configured bundle store: {path}")
    loaded = load_bundle(path, output_root=root, verify_artifacts=True)
    if not loaded.completed:
        raise BundleError("Refusing cleanup of an incomplete bundle")
    if not upload_complete(loaded):
        raise BundleError("Refusing cleanup until face and voice uploads are complete or skipped")
    shutil.rmtree(path)
