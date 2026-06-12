#!/usr/bin/env python3
"""List the OpenAI models visible to the current API key."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

_MODELS_URL = "https://api.openai.com/v1/models"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List the OpenAI models available to the API key in "
            "OPENAI_API_KEY."
        )
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Only print models whose ID contains this substring.",
    )
    parser.add_argument(
        "--owned-by",
        type=str,
        default=None,
        help="Only print models whose owned_by field matches this value.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include owner and created timestamp for each model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the filtered model list as JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def fetch_models(api_key: str, timeout_seconds: float) -> list[dict[str, Any]]:
    req = request.Request(
        _MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    with request.urlopen(req, timeout=timeout_seconds) as response:
        payload = json.load(response)

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Unexpected response shape: missing 'data' list")

    return [item for item in data if isinstance(item, dict)]


def format_created_timestamp(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    created = datetime.fromtimestamp(value, tz=timezone.utc)
    return created.isoformat()


def filter_models(
    models: list[dict[str, Any]],
    match: str | None,
    owned_by: str | None,
) -> list[dict[str, Any]]:
    filtered = models
    if match:
        filtered = [
            model
            for model in filtered
            if match.lower() in str(model.get("id", "")).lower()
        ]
    if owned_by:
        filtered = [model for model in filtered if model.get("owned_by") == owned_by]
    return sorted(filtered, key=lambda model: str(model.get("id", "")))


def main() -> int:
    args = parse_arguments()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set in the current shell.\n"
            "If it lives in ~/.bashrc, run 'source ~/.bashrc' (or open a new shell) "
            "before running this script.",
            file=sys.stderr,
        )
        return 2

    try:
        models = fetch_models(api_key=api_key, timeout_seconds=args.timeout)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        print(f"OpenAI API request failed: HTTP {exc.code}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"OpenAI API request failed: {exc.reason}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Could not parse OpenAI API response: {exc}", file=sys.stderr)
        return 1

    filtered_models = filter_models(
        models=models,
        match=args.match,
        owned_by=args.owned_by,
    )

    if args.json:
        print(json.dumps(filtered_models, indent=2, sort_keys=True))
        return 0

    if not filtered_models:
        print("No models matched the requested filters.")
        return 0

    for model in filtered_models:
        model_id = str(model.get("id", "<unknown>"))
        if not args.details:
            print(model_id)
            continue

        owner = str(model.get("owned_by", "unknown"))
        created = format_created_timestamp(model.get("created"))
        print(f"{model_id}\towner={owner}\tcreated={created}")

    print(f"\n{len(filtered_models)} model(s) visible to this API key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
