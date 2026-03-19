"""Run Step 7 — Neo4j Slide-EduKG Storage.

Reads:
  - <doc_id>_step5_concepts.json
  - <doc_id>_step5_triples_pruned.json
  - <doc_id>_step6_expansion.json
  - <doc_id>_preprocessed.json

Writes:
  - <doc_id>_step7_storage_report.json

Usage:
  python run_step7.py --doc-id CS1050-L03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 7: Neo4j storage")
    parser.add_argument("--doc-id", type=str, required=True, help="Document identifier (e.g. CS1050-L03)")
    parser.add_argument(
        "--concepts",
        type=Path,
        help="Path to concepts JSON. Defaults to <doc_id>_step5_concepts.json",
    )
    parser.add_argument(
        "--triples",
        type=Path,
        help="Path to pruned triples JSON. Defaults to <doc_id>_step5_triples_pruned.json",
    )
    parser.add_argument(
        "--expansion",
        type=Path,
        help="Path to expansion JSON. Defaults to <doc_id>_step6_expansion.json",
    )
    parser.add_argument(
        "--preprocessed",
        type=Path,
        help="Path to preprocessed JSON. Defaults to <doc_id>_preprocessed.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for report JSON. Defaults to <doc_id>_step7_storage_report.json",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        help="Neo4j URI (overrides .env setting)",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        help="Neo4j user (overrides .env setting)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        help="Neo4j password (overrides .env setting)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    backend_root = repo_root / "pace-kg" / "backend"
    sys.path.insert(0, str(backend_root))

    # Ensure .env is loaded
    os.chdir(repo_root / "pace-kg")

    from pipeline.step7_storage.extractor import store_neo4j

    concepts_path = args.concepts or (repo_root / f"{args.doc_id}_step5_concepts.json")
    triples_path = args.triples or (repo_root / f"{args.doc_id}_step5_triples_pruned.json")
    expansion_path = args.expansion or (repo_root / f"{args.doc_id}_step6_expansion.json")
    preprocessed_path = args.preprocessed or (repo_root / f"{args.doc_id}_preprocessed.json")
    output_path = args.output or (repo_root / f"{args.doc_id}_step7_storage_report.json")

    for p in (concepts_path, triples_path, expansion_path, preprocessed_path):
        if not p.exists():
            print(f"ERROR: required input file not found: {p}")
            return 1

    report = store_neo4j(
        doc_id=args.doc_id,
        concepts_path=str(concepts_path),
        extraction_triples_path=str(triples_path),
        expansion_edges_path=str(expansion_path),
        preprocessed_path=str(preprocessed_path),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
