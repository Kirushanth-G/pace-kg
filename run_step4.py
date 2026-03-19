"""Run Step 4 — LLM Triple Extraction (Groq) on existing pipeline outputs.

This script reads:
  - <doc_id>_preprocessed.json  (Step 2 output)
  - <doc_id>_keyphrases.json    (Step 3 output)

and produces:
  - <doc_id>_step4_triples.json

It mirrors the extraction logic in CLAUDE.md / pace_kg.py, including the
3-layer validation and the Groq prompt constraints.

Usage:
  python run_step4.py --doc-id CS1050-L03

If you have a valid GROQ_API_KEY, place it in pace-kg/.env (GROQ_API_KEY=...) or
set it in the environment before running.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 4: LLM triple extraction")
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
        "--output",
        type=Path,
        help="Output path for triples JSON. Defaults to <doc_id>_step4_triples.json",
    )
    parser.add_argument(
        "--groq-api-key",
        type=str,
        help="Groq API key. If omitted, reads from env var GROQ_API_KEY or .env file.",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="(Optional) limit processing to the first N slides.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    backend_root = repo_root / "pace-kg" / "backend"

    # Ensure backend modules can be imported
    sys.path.insert(0, str(backend_root))

    # Make sure config picks up .env from pace-kg/pace-kg
    os.chdir(repo_root / "pace-kg")

    if args.groq_api_key:
        os.environ["GROQ_API_KEY"] = args.groq_api_key

    # Import backend modules after setting up sys.path and cwd
    from core.config import settings
    from pipeline.step2_preprocessor.cleaner import load_preprocessed_json
    from pipeline.step4_llm_extraction import extract_triples, load_keyphrases, save_triples

    # Determine paths
    preprocessed_path = args.preprocessed or (repo_root / f"{args.doc_id}_preprocessed.json")
    keyphrases_path = args.keyphrases or (repo_root / f"{args.doc_id}_keyphrases.json")
    output_path = args.output or (repo_root / f"{args.doc_id}_step4_triples.json")

    if not preprocessed_path.exists():
        print(f"ERROR: preprocessed file not found: {preprocessed_path}")
        return 1
    if not keyphrases_path.exists():
        print(f"ERROR: keyphrases file not found: {keyphrases_path}")
        return 1

    # Load slides
    slides = load_preprocessed_json(str(preprocessed_path), args.doc_id)
    if args.max_slides is not None:
        slides = slides[: args.max_slides]

    keyphrases_by_slide = load_keyphrases(str(keyphrases_path))

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set. Provide via --groq-api-key or set env var.")
        return 1

    print(f"Running Step 4 for doc_id={args.doc_id} (slides={len(slides)})")
    triples = extract_triples(slides, keyphrases_by_slide)

    print(f"Extracted {len(triples)} validated triples.")
    save_triples(triples, str(output_path))
    print(f"Saved triples to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
