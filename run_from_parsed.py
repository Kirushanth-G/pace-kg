"""Run the PACE-KG pipeline starting from a Marker/Colab pre-parsed JSON (Step 1 output).

This script is intended to be run locally (e.g. on Windows) and does NOT require
running the Marker PDF parser. Instead it consumes an existing JSON produced by
`pace_kg.py` (Colab) or the Marker+OCR pipeline.

Usage:
    python run_from_parsed.py \
      --parsed-json CS1050-L03_parsed.json \
      --doc-id CS1050-L03

By default this script will:
  1) Load the parsed slides (Step 1) from the provided JSON.
  2) Run the Markdown preprocessor (Step 2) to produce clean slide content.
  3) Optionally run keyphrase extraction (Step 3) if sentence-transformers is
     installed.

Outputs (written next to the input JSON):
  - <doc_id>_preprocessed.json
  - <doc_id>_keyphrases.json (if Step 3 runs)

This script is a lightweight helper for working with existing pipeline artifacts
without requiring Colab/Marker.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PACE-KG pipeline from a pre-parsed JSON")
    parser.add_argument(
        "--parsed-json",
        type=Path,
        default=Path("CS1050-L03_parsed.json"),
        help="Path to the Marker/Colab pre-parsed JSON (Step 1 output).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document identifier to assign (defaults to parsed file stem without _parsed).",
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Skip keyphrase extraction (Step 3).",
    )
    args = parser.parse_args(argv)

    parsed_path = args.parsed_json
    if not parsed_path.exists():
        print(f"ERROR: Parsed JSON not found: {parsed_path}")
        return 1

    # Determine doc_id
    if args.doc_id:
        doc_id = args.doc_id
    else:
        stem = parsed_path.stem
        if stem.endswith("_parsed"):
            doc_id = stem[: -len("_parsed")]
        else:
            doc_id = stem

    # Add the backend packages to sys.path so we can import pipeline modules.
    # The backend package lives under <repo-root>/pace-kg/backend.
    repo_root = Path(__file__).resolve().parent
    backend_path = repo_root / "pace-kg" / "backend"
    sys.path.insert(0, str(backend_path))

    # Step 1: load pre-parsed slides
    from pipeline.step1_marker.parser import load_parsed_json
    from pipeline.step2_preprocessor.cleaner import preprocess_slides

    print(f"Loading parsed slides from: {parsed_path}")
    slides = load_parsed_json(parsed_path, doc_id=doc_id)
    print(f"Loaded {len(slides)} slides (doc_id={doc_id})")

    # Step 2: preprocess
    print("Running Step 2: Markdown preprocessor...")
    contents = preprocess_slides(slides)
    print(f"Step 2 complete: {len(contents)} slides processed")

    out_preprocessed = parsed_path.parent / f"{doc_id}_preprocessed.json"
    with open(out_preprocessed, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in contents], f, ensure_ascii=False, indent=2)
    print(f"Saved Step 2 output: {out_preprocessed}")

    if args.skip_step3:
        print("Skipping Step 3 (keyphrase extraction) as requested.")
        return 0

    try:
        from core.config import Settings
        from pipeline.step3_keyphrase.extractor import extract_keyphrases
    except Exception as exc:
        print("Skipping Step 3: required dependencies are not installed.")
        print(f"  Reason: {exc}")
        return 0

    print("Running Step 3: keyphrase extraction (may download models on first run) ...")
    config = Settings()
    results = {}
    for slide in contents:
        # Skip very low-content slides (e.g. title pages)
        if len(slide.clean_text.strip().split()) < 8:
            results[slide.slide_id] = []
            continue

        kps = extract_keyphrases(slide, config)
        results[slide.slide_id] = [
            {
                "phrase": kp.phrase,
                "score": kp.score,
                "source_type": kp.source_type,
                "appears_in": kp.appears_in,
            }
            for kp in kps
        ]

    out_keyphrases = parsed_path.parent / f"{doc_id}_keyphrases.json"
    with open(out_keyphrases, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved Step 3 output: {out_keyphrases}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
