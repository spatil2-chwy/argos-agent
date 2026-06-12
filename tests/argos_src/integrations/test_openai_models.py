from pathlib import Path

import pytest

from argos_src.integrations.openai_models import (
    DEFAULT_CONFIG_PATH,
    get_embeddings_model,
    get_llm_model_direct,
)


def test_default_config_path_points_to_repo_config():
    repo_root = Path(__file__).resolve().parents[3]

    assert DEFAULT_CONFIG_PATH == repo_root / "config.toml"


def test_get_llm_model_direct_rejects_unsupported_vendor_before_client_creation():
    with pytest.raises(ValueError, match="vendor='openai'"):
        get_llm_model_direct("test-model", vendor="other")


def test_get_embeddings_model_rejects_unsupported_vendor(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[vendor]
embeddings_model = "local"

[openai]
embeddings_model = "text-embedding-3-small"
""".strip()
    )

    with pytest.raises(ValueError, match="supports OpenAI embeddings"):
        get_embeddings_model(config_path)
