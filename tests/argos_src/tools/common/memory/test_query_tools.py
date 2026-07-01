from __future__ import annotations

import json

from argos_src.observability.observability import clear_request_context, set_request_context
from argos_src.tools.common.memory.query_tools import (
    SearchMemorySemanticInput,
    SearchMemorySemanticTool,
)


class _FakeMemoryProvider:
    def __init__(self) -> None:
        self.site_code = "BOS3"
        self.calls = []

    def search_semantic_memory(self, **kwargs):
        self.calls.append(("semantic", kwargs))
        return {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "transcript": "A" * 900,
                    "score": 0.92,
                    "start_time": "2026-06-01T10:00:00Z",
                    "end_time": "2026-06-01T10:05:00Z",
                    "building_code": "BOS",
                    "room_id": "lab",
                }
            ],
            "memory_items": [
                {
                    "memory_id": "memory-1",
                    "kind": "fact",
                    "key": "favorite_snack",
                    "summary": "Likes mango seltzer.",
                    "observed_at": "2026-05-01T10:00:00Z",
                    "updated_at": "2026-05-02T10:00:00Z",
                    "source": "extracted",
                    "source_ref": "episode-1",
                    "score": 0.81,
                }
            ],
        }


def _payload(raw: str) -> dict:
    return json.loads(raw)


def test_search_memory_semantic_schema_only_exposes_query_and_limit():
    properties = SearchMemorySemanticInput.model_json_schema()["properties"]

    assert set(properties) == {"query", "limit"}


def test_search_memory_semantic_defaults_to_request_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("GO2_LATENCY_CONSOLE", "0")
    monkeypatch.setenv("GO2_LATENCY_LOG_PATH", str(tmp_path / "latency.log"))
    provider = _FakeMemoryProvider()
    tool = SearchMemorySemanticTool(memory_provider=provider)
    set_request_context(owner_id="person-1")
    try:
        result = _payload(tool._run(query="snacks", limit=50))
    finally:
        clear_request_context()

    assert result["success"] is True
    assert "person_id" not in result["data"]
    assert result["data"]["memory_items"][0]["memory_id"] == "memory-1"
    assert provider.calls[0] == (
        "semantic",
        {
            "person_id": "person-1",
            "text": "snacks",
            "building_code": "BOS3",
            "limit": 50,
        },
    )


def test_person_scoped_tool_returns_error_without_owner(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("GO2_LATENCY_CONSOLE", "0")
    monkeypatch.setenv("GO2_LATENCY_LOG_PATH", str(tmp_path / "latency.log"))
    clear_request_context()
    tool = SearchMemorySemanticTool(memory_provider=_FakeMemoryProvider())

    result = _payload(tool._run(query="snacks"))

    assert result["success"] is False
    assert "No current recognized owner" in result["message"]


def test_semantic_search_includes_timestamps_and_truncated_snippet(monkeypatch, tmp_path):
    monkeypatch.setenv("GO2_LATENCY_CONSOLE", "0")
    monkeypatch.setenv("GO2_LATENCY_LOG_PATH", str(tmp_path / "latency.log"))
    provider = _FakeMemoryProvider()
    tool = SearchMemorySemanticTool(memory_provider=provider)
    set_request_context(owner_id="person-2")
    try:
        result = _payload(tool._run(query="robot demos"))
    finally:
        clear_request_context()

    episode = result["data"]["episodes"][0]
    assert episode["episode_id"] == "episode-1"
    assert episode["start_time"] == "2026-06-01T10:00:00Z"
    assert episode["building_code"] == "BOS"
    assert len(episode["snippet"]) <= 700
    assert result["data"]["memory_items"][0]["summary"] == "Likes mango seltzer."
    assert provider.calls[0][1]["person_id"] == "person-2"
