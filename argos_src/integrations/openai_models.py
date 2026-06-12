"""Small Argos model factory for OpenAI-backed LangChain models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tomli

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("rb") as handle:
        return tomli.load(handle)


def _openai_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    openai_config = dict(config.get("openai", {}))
    if not openai_config:
        raise ValueError("config.toml is missing an [openai] section.")
    return openai_config


def get_llm_model_direct(
    model_name: str,
    vendor: str = "openai",
    config_path: str | Path | None = None,
    **kwargs: Any,
):
    """Return a LangChain chat model for Argos memory extraction."""
    if vendor != "openai":
        raise ValueError(
            "Argos memory extraction currently supports vendor='openai' only."
        )
    from langchain_openai import ChatOpenAI

    openai_config = _openai_config(config_path)
    base_url = openai_config.get("base_url", "https://api.openai.com/v1/")
    logger.info("Initializing OpenAI chat model: %s", model_name)
    return ChatOpenAI(model=model_name, base_url=base_url, **kwargs)


def get_embeddings_model(
    config_path: str | Path | None = None,
    *,
    return_kwargs: bool = False,
):
    """Return the configured OpenAI embeddings model and optional save metadata."""
    from langchain_openai import OpenAIEmbeddings

    config = _load_config(config_path)
    vendor = str(config.get("vendor", {}).get("embeddings_model", "openai"))
    if vendor != "openai":
        raise ValueError(
            "Argos currently supports OpenAI embeddings for local knowledge bases. "
            f"config.toml selected embeddings_model={vendor!r}."
        )
    openai_config = _openai_config(config_path)
    model_name = openai_config.get("embeddings_model", "text-embedding-3-small")
    base_url = openai_config.get("base_url", "https://api.openai.com/v1/")
    logger.info("Initializing OpenAI embeddings model: %s", model_name)
    embeddings = OpenAIEmbeddings(model=model_name, base_url=base_url)
    if not return_kwargs:
        return embeddings
    return embeddings, {
        "class": "langchain_openai.embeddings.base.OpenAIEmbeddings",
        "model": model_name,
        "base_url": base_url,
        "vendor": "openai",
    }
