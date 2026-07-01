"""Read-only Tailwag memory query tools for realtime agent use."""

from __future__ import annotations

from typing import Any, Iterable, Type

from pydantic import BaseModel, ConfigDict, Field

from argos_src.observability.observability import (
    LatencyLogger,
    get_request_context,
    perf_now,
)
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json


MAX_TEXT_CHARS = 700
MAX_LIMIT = 20


class SearchMemorySemanticInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language query for semantic search over remembered episodes "
            "and extracted memory items."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_LIMIT,
        description="Maximum number of results to return.",
    )


class SearchMemorySemanticTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search_memory_semantic"
    description: str = (
        "Search Tailwag memory semantically for the current person, including both "
        "raw remembered episode transcripts and extracted durable memory items. "
        "Use this when the user asks about prior conversations, remembered facts, "
        "or evidence from memory that is not already in the prompt."
    )
    args_schema: Type[BaseModel] = SearchMemorySemanticInput
    memory_provider: Any = Field(exclude=True)
    latency_logger: LatencyLogger = Field(
        default_factory=lambda: LatencyLogger("tool"),
        exclude=True,
    )

    @staticmethod
    def _current_owner_id() -> str:
        ctx = get_request_context()
        return str(ctx.get("owner_id", "") or "").strip()

    def _site_code(self) -> str | None:
        rendered = str(getattr(self.memory_provider, "site_code", "") or "").strip()
        return rendered or None

    @staticmethod
    def _missing_owner_response() -> str:
        return tool_response_json(
            success=False,
            status="error",
            message=(
                "No current recognized owner is available for person-scoped memory."
            ),
            result_source="immediate",
        )

    def _emit_query_start(self, **fields: Any) -> float:
        started = perf_now()
        self.latency_logger.emit(
            event="memory_query_start",
            tool=self.name,
            query_kind="semantic",
            **fields,
        )
        return started

    def _emit_query_result(
        self,
        started: float,
        *,
        result_count: int,
        status: str = "completed",
        **fields: Any,
    ) -> None:
        self.latency_logger.timing(
            "memory_query_s",
            perf_now() - started,
            tool=self.name,
            query_kind="semantic",
            result_count=result_count,
            status=status,
            **fields,
        )

    def _run(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        resolved_person_id = self._current_owner_id()
        if not resolved_person_id:
            return self._missing_owner_response()
        started = self._emit_query_start(person_id=resolved_person_id)
        try:
            results = self.memory_provider.search_semantic_memory(
                text=query,
                person_id=resolved_person_id,
                building_code=self._site_code(),
                limit=limit,
            )
        except Exception as exc:
            self._emit_query_result(started, result_count=0, status="error")
            return _error_response(str(exc))
        episode_entries = [_episode_entry(item) for item in results.get("episodes", [])]
        memory_entries = [
            _memory_item_entry(item) for item in results.get("memory_items", [])
        ]
        result_count = len(episode_entries) + len(memory_entries)
        self._emit_query_result(
            started,
            result_count=result_count,
            person_id=resolved_person_id,
        )
        return tool_response_json(
            success=True,
            status="completed",
            message=(
                f"Found {len(episode_entries)} matching episode(s) and "
                f"{len(memory_entries)} matching memory item(s)."
            ),
            result_source="tailwag",
            data={
                "query": query,
                "episodes": episode_entries,
                "memory_items": memory_entries,
            },
        )


def get_memory_query_tools(
    memory_provider: Any,
    runtime_tool_names: Iterable[str] | None = None,
) -> list[BaseTool]:
    """Build selected read-only Tailwag memory query tools."""
    if memory_provider is None:
        return []
    requested = (
        None
        if runtime_tool_names is None
        else {str(name or "").strip() for name in runtime_tool_names}
    )
    candidates: list[BaseTool] = [
        SearchMemorySemanticTool(memory_provider=memory_provider),
    ]
    if requested is None:
        return candidates
    return [tool for tool in candidates if getattr(tool, "name", "") in requested]


def _episode_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": str(item.get("episode_id") or ""),
        "start_time": str(item.get("start_time") or "").strip() or None,
        "end_time": str(item.get("end_time") or "").strip() or None,
        "building_code": str(item.get("building_code") or "").strip() or None,
        "room_id": str(item.get("room_id") or "").strip() or None,
        "score": item.get("score"),
        "snippet": _truncate_text(item.get("transcript")),
    }


def _memory_item_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": str(item.get("memory_id") or ""),
        "kind": str(item.get("kind") or ""),
        "key": str(item.get("key") or ""),
        "summary": _truncate_text(item.get("summary")),
        "observed_at": str(item.get("observed_at") or "").strip() or None,
        "updated_at": str(item.get("updated_at") or "").strip() or None,
        "source": str(item.get("source") or "").strip() or None,
        "source_ref": str(item.get("source_ref") or "").strip() or None,
        "score": item.get("score"),
    }


def _error_response(message: str) -> str:
    return tool_response_json(
        success=False,
        status="error",
        message=message,
        result_source="tailwag",
    )


def _truncate_text(value: Any, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    rendered = " ".join(str(value or "").split())
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 3)].rstrip() + "..."
