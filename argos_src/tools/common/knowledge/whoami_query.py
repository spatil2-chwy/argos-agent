"""Built-in knowledge-base tool builders for profile-configured knowledge tools."""

from __future__ import annotations

from typing import Optional

from argos_src.profile_config import KnowledgeBaseProfile, resolve_repo_path
from argos_src.observability.observability import LatencyLogger, perf_now
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json

SUPPORTED_KNOWLEDGE_BASE_KINDS = {"whoami_query"}


def build_knowledge_tool(entry: KnowledgeBaseProfile) -> Optional[BaseTool]:
    """Build one built-in knowledge tool from profile config."""
    if entry.kind not in SUPPORTED_KNOWLEDGE_BASE_KINDS:
        raise ValueError(
            f"Unsupported knowledge base kind '{entry.kind}'. "
            f"Supported kinds: {', '.join(sorted(SUPPORTED_KNOWLEDGE_BASE_KINDS))}."
        )

    root_dir = resolve_repo_path(entry.root_dir)
    generated_path = root_dir / "generated"
    if not generated_path.exists() or not (generated_path / "index.faiss").exists():
        print(f"Knowledge base '{entry.tool_name}' not found at {root_dir}.")
        return None

    try:
        from argos_src.knowledge import QueryKnowledgeBaseTool

        class TimedKnowledgeTool(QueryKnowledgeBaseTool):
            def _run(self, query: str) -> str:
                latency_logger = LatencyLogger("tool")
                t0 = perf_now()
                try:
                    result = super()._run(query)
                    latency_logger.timing(
                        "tool_run_s",
                        perf_now() - t0,
                        tool=self.name,
                    )
                    return tool_response_json(
                        success=True,
                        status="completed",
                        message="Knowledge lookup completed.",
                        result_source="immediate",
                        data={
                            "query": query,
                            "answer": result,
                        },
                    )
                except Exception as exc:
                    latency_logger.timing(
                        "tool_run_s",
                        perf_now() - t0,
                        tool=self.name,
                        status="error",
                    )
                    return tool_response_json(
                        success=False,
                        status="error",
                        message=str(exc),
                        result_source="immediate",
                        data={"query": query},
                    )

        tool = TimedKnowledgeTool(root_dir=str(root_dir), k=int(entry.k))
        tool.name = entry.tool_name
        tool.description = entry.description
        print(f"Knowledge base '{entry.tool_name}' loaded from {root_dir}")
        return tool
    except Exception as exc:
        print(f"Failed to load knowledge base '{entry.tool_name}': {exc}")
        return None


def get_default_chewy_knowledge_profile() -> KnowledgeBaseProfile:
    """Return the legacy Chewy knowledge-base config."""
    return KnowledgeBaseProfile(
        kind="whoami_query",
        root_dir="chewy_docs",
        tool_name="query_chewy_knowledge",
        description=(
            "Search Chewy Robotics company documentation and knowledge base. "
            "Use this for any informational question about Chewy Robotics "
            "products, team, hiring, strategy, policies, pricing, or anything company-related."
        ),
        k=4,
    )
