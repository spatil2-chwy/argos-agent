"""FAISS-backed local knowledge bases for Argos."""

from __future__ import annotations

import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

from argos_src.integrations.openai_models import get_embeddings_model
from argos_src.tools.base import BaseTool


class QueryKnowledgeBaseToolInput(BaseModel):
    query: str = Field(..., description="The query to search the knowledge base with")


def _class_from_string(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def _initialize_embeddings(class_path: str, kwargs: dict[str, Any]) -> Any:
    cls = _class_from_string(class_path)
    constructor_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(cls).parameters
    }
    return cls(**constructor_kwargs)


def _load_saved_embeddings(generated_dir: Path) -> Any:
    kwargs_path = generated_dir / "vdb_kwargs.json"
    if not kwargs_path.exists():
        return get_embeddings_model()
    with kwargs_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    embedding_kwargs = dict(saved.get("embeddings") or {})
    class_path = str(
        embedding_kwargs.pop(
            "class",
            "langchain_openai.embeddings.base.OpenAIEmbeddings",
        )
    )
    embedding_kwargs.pop("vendor", None)
    return _initialize_embeddings(class_path, embedding_kwargs)


def load_faiss_client(
    generated_dir: str | Path,
    embeddings_model: Any | None = None,
) -> Any:
    """Load a FAISS vector store from an Argos or legacy generated directory."""
    from langchain_community.vectorstores import FAISS

    generated_path = Path(generated_dir)
    embeddings = embeddings_model or _load_saved_embeddings(generated_path)
    return FAISS.load_local(
        folder_path=generated_path.as_posix(),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


class QueryKnowledgeBaseTool(BaseTool):
    name: str = "query_database"
    description: str = "Query the knowledge base with a natural language query"
    args_schema: Type[QueryKnowledgeBaseToolInput] = QueryKnowledgeBaseToolInput

    root_dir: str = Field(..., description="The knowledge base root directory")
    k: int = Field(default=4, description="The number of results to return")
    embeddings_model: Any | None = None
    _vdb_client: Any | None = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vdb_client = load_faiss_client(
            Path(self.root_dir) / "generated",
            self.embeddings_model,
        )

    def _run(self, query: str) -> str:
        if self._vdb_client is None:
            raise RuntimeError("Knowledge base client is not initialized.")
        return str(self._vdb_client.similarity_search(query, k=int(self.k)))


def build_faiss_knowledge_base(
    root_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    embeddings_model: Any | None = None,
) -> Any:
    """Build a FAISS index from files under a knowledge base documentation folder."""
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    root_path = Path(root_dir)
    generated_dir = Path(output_dir) if output_dir is not None else root_path / "generated"
    documentation_dir = root_path / "documentation"
    if not documentation_dir.exists():
        raise FileNotFoundError(f"Missing documentation directory: {documentation_dir}")

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".doc": Docx2txtLoader,
        ".docx": Docx2txtLoader,
        ".urdf": TextLoader,
        ".xacro": TextLoader,
    }
    documents = []
    for path in sorted(documentation_dir.rglob("*")):
        if not path.is_file():
            continue
        loader_cls = loaders.get(path.suffix.lower())
        if loader_cls is None:
            continue
        documents.extend(loader_cls(file_path=path.as_posix()).load())

    urdf_dir = root_path / "urdfs"
    if urdf_dir.exists():
        for path in sorted(urdf_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".urdf", ".xacro"}:
                documents.extend(TextLoader(file_path=path.as_posix()).load())

    if not documents:
        raise ValueError(f"No supported documents found under {documentation_dir}")

    embeddings = embeddings_model
    embedding_kwargs: dict[str, Any]
    if embeddings is None:
        embeddings, embedding_kwargs = get_embeddings_model(return_kwargs=True)
    else:
        embedding_kwargs = {
            "class": str(embeddings.__class__)
            .strip("<>")
            .replace("class '", "")
            .replace("'", ""),
        }

    generated_dir.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(generated_dir.as_posix())
    metadata = {
        "vectorstore": {"class": "langchain_community.vectorstores.faiss.FAISS"},
        "embeddings": embedding_kwargs,
    }
    (generated_dir / "vdb_kwargs.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    info_path = generated_dir / "info.json"
    if not info_path.exists():
        info_path.write_text("{}", encoding="utf-8")
    return db
