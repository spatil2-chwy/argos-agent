"""Argos-owned knowledge-base tooling."""

from .faiss_store import QueryKnowledgeBaseTool, build_faiss_knowledge_base

__all__ = ["QueryKnowledgeBaseTool", "build_faiss_knowledge_base"]
