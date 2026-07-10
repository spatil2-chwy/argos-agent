#!/usr/bin/env python3
"""Compare live Argos voice embeddings against Tailwag/Neo4j voice scores."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argos_src.speaker_recognition.backend import SpeechBrainEcapaBackend
from scripts.labs.speaker_lab_common import (
    load_audio_file_as_agent_pcm16,
    render_stats_payload,
)


VOICE_THRESHOLD = 0.50
VOICE_MARGIN_THRESHOLD = 0.20


def _neo4j_cosine_to_raw(score: Any) -> float:
    return (float(score or 0.0) * 2.0) - 1.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed one saved WAV with the Argos ECAPA backend and compare Tailwag "
            "search, Neo4j vector-index scores, and direct cosine scores."
        )
    )
    parser.add_argument("--audio-file", required=True, help="WAV file to score")
    parser.add_argument("--site-code", default="", help="optional Tailwag site filter")
    parser.add_argument("--limit", type=int, default=10, help="maximum rows to return")
    parser.add_argument(
        "--person-id",
        default="",
        help="optional person id to highlight in direct-reference rows",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="optional path to write the report JSON",
    )
    return parser


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32).reshape(-1)))


def _normalize(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = _norm(array)
    if norm <= 1e-8:
        return np.asarray([], dtype=np.float32)
    return array / norm


def _cosine(left: Any, right: Any) -> float:
    lhs = _normalize(left)
    rhs = _normalize(right)
    if lhs.size <= 0 or rhs.size <= 0 or lhs.size != rhs.size:
        return 0.0
    return float(np.dot(lhs, rhs))


def _load_tailwag() -> tuple[Any, Any, Any]:
    try:
        from tailwag_memory import TailwagMemoryClient, load_settings
        from tailwag_memory.db import Neo4jQueryRunner
    except Exception as exc:
        raise RuntimeError(
            "tailwag-memory is required. Run from an Argos shell where "
            "`python3 -m pip install -e ../tailwag-memory` has been done."
        ) from exc
    settings = load_settings()
    runner = Neo4jQueryRunner(settings)
    client = TailwagMemoryClient(runner, settings)
    return client, runner, settings


def _tailwag_search(
    client: Any,
    *,
    embedding: list[float],
    site_code: str,
    limit: int,
) -> dict[str, Any]:
    result = client.search_voice(
        embedding=embedding,
        limit=max(1, int(limit)),
        site_code=site_code or None,
    )
    return _plain(result)


def _neo4j_vector_rows(
    runner: Any,
    *,
    embedding: list[float],
    site_code: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = runner.run(
        """
        CALL db.index.vector.queryNodes('voice_reference_embedding', $candidate_limit, $embedding)
        YIELD node AS ref, score
        WHERE ref:VoiceReference
        MATCH (person:Person)-[:HAS_VOICE_REFERENCE]->(ref)
        OPTIONAL MATCH (person)-[:HAS_DIRECTORY_RECORD]->(directory:EmployeeDirectoryRecord)
        WHERE coalesce(ref.status, 'active') = 'active'
          AND coalesce(person.status, 'active') <> 'archived'
          AND coalesce(ref.consent_status, person.consent_status, '') = 'consented'
          AND ($site_code IS NULL OR directory IS NULL OR directory.site_code = $site_code)
        RETURN person.id AS person_id,
               person.display_name AS display_name,
               ref.id AS reference_id,
               ref.model AS model,
               coalesce(ref.sample_count, 1) AS sample_count,
               coalesce(ref.target_sample_count, 0) AS target_sample_count,
               score AS neo4j_score
        ORDER BY neo4j_score DESC
        LIMIT $limit
        """,
        {
            "candidate_limit": max(max(1, int(limit)) * 5, 25),
            "limit": max(1, int(limit)),
            "embedding": embedding,
            "site_code": str(site_code or "").strip() or None,
        },
    )
    rendered_rows: list[dict[str, Any]] = []
    for row in rows:
        rendered = dict(row)
        rendered["raw_cosine_estimate"] = round(
            _neo4j_cosine_to_raw(rendered.get("neo4j_score")),
            6,
        )
        rendered_rows.append(rendered)
    return rendered_rows


def _reference_rows(
    runner: Any,
    *,
    site_code: str,
) -> list[dict[str, Any]]:
    rows = runner.run(
        """
        MATCH (person:Person)-[:HAS_VOICE_REFERENCE]->(ref:VoiceReference)
        OPTIONAL MATCH (person)-[:HAS_DIRECTORY_RECORD]->(directory:EmployeeDirectoryRecord)
        WHERE coalesce(ref.status, 'active') = 'active'
          AND coalesce(person.status, 'active') <> 'archived'
          AND coalesce(ref.consent_status, person.consent_status, '') = 'consented'
          AND ($site_code IS NULL OR directory IS NULL OR directory.site_code = $site_code)
        RETURN person.id AS person_id,
               person.display_name AS display_name,
               ref.id AS reference_id,
               ref.model AS model,
               ref.embedding AS embedding,
               coalesce(ref.sample_count, 1) AS sample_count,
               coalesce(ref.target_sample_count, 0) AS target_sample_count
        """,
        {"site_code": str(site_code or "").strip() or None},
    )
    return [dict(row) for row in rows]


def _direct_cosine_rows(
    *,
    query_embedding: np.ndarray,
    references: list[dict[str, Any]],
    limit: int,
    highlight_person_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in references:
        reference_embedding = row.get("embedding") or []
        cosine = _cosine(query_embedding, reference_embedding)
        rows.append(
            {
                "person_id": str(row.get("person_id") or ""),
                "display_name": str(row.get("display_name") or ""),
                "reference_id": str(row.get("reference_id") or ""),
                "model": str(row.get("model") or ""),
                "sample_count": int(row.get("sample_count") or 0),
                "target_sample_count": int(row.get("target_sample_count") or 0),
                "direct_cosine": round(cosine, 6),
                "cosine_mapped_0_1": round((cosine + 1.0) / 2.0, 6),
                "embedding_norm": round(_norm(np.asarray(reference_embedding, dtype=np.float32)), 6),
                "highlighted": bool(
                    highlight_person_id
                    and str(row.get("person_id") or "") == highlight_person_id
                ),
            }
        )
    rows.sort(key=lambda item: float(item["direct_cosine"]), reverse=True)
    return rows[: max(1, int(limit))]


def _decision_summary(search: dict[str, Any], direct_rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = direct_rows[0] if direct_rows else {}
    runner_up = direct_rows[1] if len(direct_rows) > 1 else {}
    direct_top = float(top.get("direct_cosine", 0.0) or 0.0)
    direct_runner_up = float(runner_up.get("direct_cosine", 0.0) or 0.0)
    direct_margin = max(0.0, direct_top - direct_runner_up)
    tailwag_candidates = search.get("candidates") or []
    tailwag_top = tailwag_candidates[0] if tailwag_candidates else {}
    tailwag_top_score = float(search.get("top_score", 0.0) or 0.0)
    tailwag_runner_up_score = float(search.get("runner_up_score", 0.0) or 0.0)
    has_tailwag_runner_up = len(tailwag_candidates) > 1
    return {
        "tailwag_status": search.get("status"),
        "tailwag_reason": search.get("reason"),
        "tailwag_recognized": bool(search.get("recognized")),
        "tailwag_top_person_id": tailwag_top.get("person_id"),
        "tailwag_top_score": tailwag_top_score,
        "tailwag_top_score_raw_cosine_estimate": round(
            _neo4j_cosine_to_raw(tailwag_top_score),
            6,
        ),
        "tailwag_runner_up_score": tailwag_runner_up_score,
        "tailwag_runner_up_score_raw_cosine_estimate": (
            round(_neo4j_cosine_to_raw(tailwag_runner_up_score), 6)
            if has_tailwag_runner_up
            else None
        ),
        "tailwag_margin": search.get("margin"),
        "tailwag_margin_raw_cosine_estimate": (
            round(
                _neo4j_cosine_to_raw(tailwag_top_score)
                - _neo4j_cosine_to_raw(tailwag_runner_up_score),
                6,
            )
            if has_tailwag_runner_up
            else None
        ),
        "tailwag_threshold": search.get("threshold"),
        "tailwag_threshold_raw_cosine_estimate": round(
            _neo4j_cosine_to_raw(search.get("threshold")),
            6,
        ),
        "tailwag_margin_threshold": search.get("margin_threshold"),
        "direct_top_person_id": top.get("person_id"),
        "direct_top_cosine": round(direct_top, 6),
        "direct_runner_up_cosine": round(direct_runner_up, 6),
        "direct_margin": round(direct_margin, 6),
        "single_reference_margin_warning": len(direct_rows) == 1,
        "note": (
            "Neo4j cosine vector-index scores are on a 0..1 scale. Convert them back "
            "to raw cosine with raw = (score * 2) - 1 before comparing against eval "
            "thresholds. With only one active voice reference, Tailwag's runner-up "
            "score is 0, so the margin gate cannot distinguish impostors; only the "
            "top-score threshold is protecting the identity."
            if len(direct_rows) == 1
            else (
                "Neo4j cosine vector-index scores are on a 0..1 scale. Convert them "
                "back to raw cosine with raw = (score * 2) - 1 before comparing "
                "against eval thresholds."
            )
        ),
    }


def main() -> int:
    args = _build_parser().parse_args()
    audio_pcm16, audio_meta = load_audio_file_as_agent_pcm16(args.audio_file)
    backend = SpeechBrainEcapaBackend()
    waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
    query_embedding = backend.embed_query_clip(waveform, sample_rate=16000)
    embedding_list = [float(value) for value in query_embedding.tolist()]

    client, runner, settings = _load_tailwag()
    site_code = str(args.site_code or "").strip()
    search = _tailwag_search(
        client,
        embedding=embedding_list,
        site_code=site_code,
        limit=int(args.limit),
    )
    vector_rows = _neo4j_vector_rows(
        runner,
        embedding=embedding_list,
        site_code=site_code,
        limit=int(args.limit),
    )
    reference_rows = _reference_rows(runner, site_code=site_code)
    direct_rows = _direct_cosine_rows(
        query_embedding=query_embedding,
        references=reference_rows,
        limit=int(args.limit),
        highlight_person_id=str(args.person_id or "").strip(),
    )
    report = {
        "audio_file": str(Path(args.audio_file).expanduser().resolve()),
        "audio_meta": audio_meta,
        "audio_stats": render_stats_payload(audio_pcm16),
        "query_embedding": {
            "model": getattr(backend, "model_name", "speechbrain_ecapa"),
            "dimension": len(embedding_list),
            "norm": round(_norm(query_embedding), 6),
        },
        "tailwag_settings": {
            "site_code_filter": site_code or None,
            "configured_voice_embedding_model": getattr(settings, "voice_embedding_model", ""),
            "configured_voice_embedding_dimension": getattr(settings, "voice_embedding_dimension", 0),
            "default_voice_threshold": VOICE_THRESHOLD,
            "default_voice_margin_threshold": VOICE_MARGIN_THRESHOLD,
        },
        "decision_summary": _decision_summary(search, direct_rows),
        "tailwag_search": search,
        "neo4j_vector_index_rows": vector_rows,
        "direct_cosine_rows": direct_rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
