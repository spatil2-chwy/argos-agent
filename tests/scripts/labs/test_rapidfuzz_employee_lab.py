from __future__ import annotations

from typing import Any

from scripts.labs.rapidfuzz_employee_lab import (
    LoadedDirectory,
    SpokenName,
    parse_site_codes,
    process_transcript,
    resolve_against_directories,
)


class _FakeDirectoryService:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def resolve_identity(
        self,
        shared_first_name: str = "",
        shared_last_name: str = "",
        shared_name: str = "",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "shared_first_name": shared_first_name,
                "shared_last_name": shared_last_name,
                "shared_name": shared_name,
            }
        )
        return self.result

    def shutdown(self) -> None:
        pass


def test_parse_site_codes_accepts_commas_spaces_and_deduplicates() -> None:
    assert parse_site_codes(["bos1,bos3", "BOS1", " bos2 "]) == (
        "BOS1",
        "BOS3",
        "BOS2",
    )


def test_parse_site_codes_uses_profile_fallback() -> None:
    assert parse_site_codes([], fallback_site="bos3") == ("BOS3",)


def test_resolve_against_directories_keeps_site_on_candidates() -> None:
    result = {
        "success": True,
        "status": "single_match",
        "data": {
            "candidates": [
                {
                    "official_name": "Sakshee Patil",
                    "business_title": "AI Technologist II",
                    "match_score": 100.0,
                }
            ]
        },
    }
    service = _FakeDirectoryService(result)
    directory = LoadedDirectory(
        site_code="BOS3",
        service=service,  # type: ignore[arg-type]
        record_count=1,
    )

    payload = resolve_against_directories(
        SpokenName(
            shared_name="Sakshi Patil",
            shared_first_name="Sakshi",
            shared_last_name="Patil",
        ),
        [directory],
    )

    assert service.calls == [
        {
            "shared_first_name": "Sakshi",
            "shared_last_name": "Patil",
            "shared_name": "Sakshi Patil",
        }
    ]
    assert payload["best_candidates"] == [
        {
            "official_name": "Sakshee Patil",
            "business_title": "AI Technologist II",
            "match_score": 100.0,
            "site_code": "BOS3",
            "site_status": "single_match",
        }
    ]


def test_process_transcript_adds_hint_when_agent_probe_returns_empty_name() -> None:
    result = {
        "success": True,
        "status": "no_match",
        "data": {"candidates": []},
    }
    directory = LoadedDirectory(
        site_code="BOS3",
        service=_FakeDirectoryService(result),  # type: ignore[arg-type]
        record_count=1,
    )

    payload = process_transcript(
        "hello there",
        SpokenName(shared_name="", shared_first_name="", shared_last_name=""),
        [directory],
        agent_probe={"success": True, "arguments": {}},
    )

    assert "empty name fields" in payload["diagnostic_hint"]
    assert payload["name_extraction_mode"] == "agent"


def test_process_transcript_includes_agent_probe_and_candidates() -> None:
    result = {
        "success": True,
        "status": "single_match",
        "data": {
            "candidates": [
                {
                    "official_name": "Sakshee Patil",
                    "business_title": "AI Technologist II",
                    "match_score": 100.0,
                }
            ]
        },
    }
    directory = LoadedDirectory(
        site_code="BOS3",
        service=_FakeDirectoryService(result),  # type: ignore[arg-type]
        record_count=1,
    )

    payload = process_transcript(
        "my name is sakshi patil",
        SpokenName(
            shared_name="Sakshi Patil",
            shared_first_name="Sakshi",
            shared_last_name="Patil",
        ),
        [directory],
        agent_probe={
            "success": True,
            "tool_name": "resolve_employee_identity",
            "arguments": {
                "shared_name": "Sakshi Patil",
                "shared_first_name": "Sakshi",
                "shared_last_name": "Patil",
            },
        },
    )

    assert payload["spoken_name"] == {
        "shared_name": "Sakshi Patil",
        "shared_first_name": "Sakshi",
        "shared_last_name": "Patil",
    }
    assert payload["best_candidates"][0]["official_name"] == "Sakshee Patil"
    assert payload["agent_probe"]["tool_name"] == "resolve_employee_identity"
