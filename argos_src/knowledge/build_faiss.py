"""Command-line builder for Argos FAISS knowledge bases."""

from __future__ import annotations

import argparse
from pathlib import Path

from argos_src.knowledge.faiss_store import build_faiss_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Argos FAISS knowledge base from documentation files.",
    )
    parser.add_argument("root_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for index.faiss/index.pkl/vdb_kwargs.json. Defaults to ROOT/generated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_faiss_knowledge_base(args.root_dir, output_dir=args.output_dir)
    output_dir = args.output_dir or args.root_dir / "generated"
    print(f"Built FAISS knowledge base at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
