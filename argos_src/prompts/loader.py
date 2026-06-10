"""System prompt loading for the Argos realtime agent."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _PROMPTS_DIR / "static_interaction_prompt.md"


def _with_existing_markdown_or_text(path: Path) -> Path:
    """Prefer an existing .md/.txt sibling when callers use the old suffix."""
    if path.exists():
        return path
    if path.suffix.lower() == ".txt":
        markdown_path = path.with_suffix(".md")
        if markdown_path.exists():
            return markdown_path
    if path.suffix.lower() == ".md":
        text_path = path.with_suffix(".txt")
        if text_path.exists():
            return text_path
    return path


def resolve_prompt_path(path: Path | str | None = None) -> Path:
    """Resolve a prompt path.

    Bare filenames are first looked up under argos_src/prompts/ so callers can
    pass values like ``static_interaction_prompt.md`` without a long path.
    """
    if path is None:
        return _PROMPT_PATH

    candidate = Path(path)
    if candidate.is_absolute():
        return _with_existing_markdown_or_text(candidate)
    if candidate.parent == Path("."):
        prompt_dir_candidate = _PROMPTS_DIR / candidate
        resolved_candidate = _with_existing_markdown_or_text(prompt_dir_candidate)
        if resolved_candidate.exists():
            return resolved_candidate
    return _with_existing_markdown_or_text(candidate.resolve())


def load_system_prompt(path: Path | str | None = None) -> str:
    """Load an Argos system prompt from disk."""
    return resolve_prompt_path(path).read_text().strip()


def get_prompt_path() -> Path:
    """Return the default path to the system prompt file."""
    return _PROMPT_PATH
