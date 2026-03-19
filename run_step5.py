"""Run Step 5 — Concept Weighting & Pruning.

Reads:
  - <doc_id>_preprocessed.json  (Step 2)
  - <doc_id>_keyphrases.json    (Step 3)
  - <doc_id>_step4_triples.json (Step 4)

Writes:
  - <doc_id>_step5_concepts.json
  - <doc_id>_step5_triples_pruned.json

Usage:
  python run_step5.py --doc-id CS1050-L03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 5: Concept Weighting & Pruning")
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
        help="Path to triples JSON (Step 4). Defaults to <doc_id>_step4_triples.json",
    )
    parser.add_argument(
        "--output-concepts",
        type=Path,
        help="Output path for concepts JSON. Defaults to <doc_id>_step5_concepts.json",
    )
    parser.add_argument(
        "--output-triples",
        type=Path,
        help="Output path for pruned triples JSON. Defaults to <doc_id>_step5_triples_pruned.json",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    backend_root = repo_root / "pace-kg" / "backend"
    sys.path.insert(0, str(backend_root))

    # Ensure .env is loaded by core.config
    os.chdir(repo_root / "pace-kg")

    from pipeline.step2_preprocessor.cleaner import load_preprocessed_json
    from pipeline.step5_weighting.extractor import (
        run_weighting,
        save_concepts,
        save_triples,
    )

    preprocessed_path = args.preprocessed or (repo_root / f"{args.doc_id}_preprocessed.json")
    keyphrases_path = args.keyphrases or (repo_root / f"{args.doc_id}_keyphrases.json")
    triples_path = args.triples or (repo_root / f"{args.doc_id}_step4_triples.json")

    output_concepts = args.output_concepts or (repo_root / f"{args.doc_id}_step5_concepts.json")
    output_triples = args.output_triples or (repo_root / f"{args.doc_id}_step5_triples_pruned.json")

    for p in (preprocessed_path, keyphrases_path, triples_path):
        if not p.exists():
            print(f"ERROR: required input file not found: {p}")
            return 1

    slides = load_preprocessed_json(str(preprocessed_path), args.doc_id)

    from pipeline.step5_weighting.extractor import load_keyphrases, load_triples

    keyphrases = load_keyphrases(str(keyphrases_path))
    triples = load_triples(str(triples_path))

    concept_nodes, pruned_triples = run_weighting(
        slides=slides,
        keyphrases_by_slide=keyphrases,
        triples=triples,
        doc_id=args.doc_id,
    )

    save_concepts(concept_nodes, str(output_concepts))
    save_triples(pruned_triples, str(output_triples))

    print(f"Saved concepts: {output_concepts}")
    print(f"Saved pruned triples: {output_triples}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
