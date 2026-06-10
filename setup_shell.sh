#!/usr/bin/env sh

if [ -n "${BASH_SOURCE:-}" ]; then
    SCRIPT_PATH="$BASH_SOURCE"
else
    SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
POETRY_ENV_PATH="$(cd "$SCRIPT_DIR" && poetry env info --path 2>/dev/null)"

if [ -z "$POETRY_ENV_PATH" ] || [ ! -f "$POETRY_ENV_PATH/bin/activate" ]; then
    echo "Failed to locate the Argos Poetry environment. Run 'cd ~/argos-agent && poetry install' first." >&2
    return 1 2>/dev/null || exit 1
fi

# Suppress ShellCheck warning about not following external file.
# shellcheck disable=SC1091
. "$POETRY_ENV_PATH/bin/activate"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
