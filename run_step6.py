"""Run Step 6 — Closed-Corpus Concept Expansion.

Reads:
  - <doc_id>_preprocessed.json  (Step 2)
  - <doc_id>_keyphrases.json    (Step 3)
  - <doc_id>_step5_triples_pruned.json (Step 5)

Writes:
  - <doc_id>_step6_expansion.json

Usage:
  python run_step6.py --doc-id CS1050-L03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 6: Closed-corpus expansion")
    parser.add_argument("--doc-id", type=str, required=True, help="Document identifier (e.g. CS1050-L03)")
    parser.add_argument(
        "--preprocessed",
        type=Path,
        help="Path to preprocessed JSON (Step 2). Defaults to <doc_id>_preprocessed.json",
    )
    parser.add_argument(
        "--keyphrases",
        type=Path,
        help="Path to keyphrases JSON (Step 3). Defaults to <doc_id>_keyphrases.json",
    )
    parser.add_argument(
        "--triples",
        type=Path,
        help="Path to pruned triples JSON (Step 5). Defaults to <doc_id>_step5_triples_pruned.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for expansion JSON. Defaults to <doc_id>_step6_expansion.json",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    backend_root = repo_root / "pace-kg" / "backend"
    sys.path.insert(0, str(backend_root))

    # Ensure .env is loaded
    os.chdir(repo_root / "pace-kg")

    from pipeline.step2_preprocessor.cleaner import load_preprocessed_json
    from pipeline.step6_expansion.extractor import run_expansion, save_expansion_edges

    preprocessed_path = args.preprocessed or (repo_root / f"{args.doc_id}_preprocessed.json")
    keyphrases_path = args.keyphrases or (repo_root / f"{args.doc_id}_keyphrases.json")
    triples_path = args.triples or (repo_root / f"{args.doc_id}_step5_triples_pruned.json")
    output_path = args.output or (repo_root / f"{args.doc_id}_step6_expansion.json")

    for p in (preprocessed_path, keyphrases_path, triples_path):
        if not p.exists():
            print(f"ERROR: required input file not found: {p}")
            return 1

    slides = load_preprocessed_json(str(preprocessed_path), args.doc_id)

    keyphrases = None
    with open(keyphrases_path, "r", encoding="utf-8") as f:
        keyphrases = json.load(f)

    triples = None
    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    edges = run_expansion(
        slides=slides,
        keyphrases_by_slide=keyphrases,
        pruned_triples=triples,
        doc_id=args.doc_id,
    )

    save_expansion_edges(edges, str(output_path))

    print(f"Saved expansion edges: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
