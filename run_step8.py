#!/usr/bin/env python3
"""Run Step 8 — LM-EduKG Aggregation (standalone)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 8 LM-EduKG aggregation")
    parser.add_argument("--doc-id", type=str, required=True, help="Document identifier (e.g. CS1050-L03)")
    parser.add_argument("--concepts", type=Path, help="Path to step5 concepts JSON")
    parser.add_argument("--triples", type=Path, help="Path to step5 triples JSON")
    parser.add_argument("--expansion", type=Path, help="Path to step6 expansion JSON")
    parser.add_argument("--preprocessed", type=Path, help="Path to step2 preprocessed JSON")
    parser.add_argument("--step7-report", type=Path, help="Path to step7 report JSON")
    parser.add_argument("--output", type=Path, help="Output path for LM-EduKG JSON")
    parser.add_argument("--summary-output", type=Path, help="Output path for summary JSON")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI (overrides .env setting)")
    parser.add_argument("--neo4j-user", type=str, help="Neo4j user (overrides .env setting)")
    parser.add_argument("--neo4j-password", type=str, help="Neo4j password (overrides .env setting)")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    backend_root = repo_root / "pace-kg" / "backend"
    sys.path.insert(0, str(backend_root))

    # Ensure .env is loaded
    os.chdir(repo_root / "pace-kg")

    from pipeline.step8_aggregation.extractor import run_aggregation

    concepts_path = args.concepts or (repo_root / f"{args.doc_id}_step5_concepts.json")
    triples_path = args.triples or (repo_root / f"{args.doc_id}_step5_triples_pruned.json")
    expansion_path = args.expansion or (repo_root / f"{args.doc_id}_step6_expansion.json")
    preprocessed_path = args.preprocessed or (repo_root / f"{args.doc_id}_preprocessed.json")
    step7_report_path = args.step7_report or (repo_root / f"{args.doc_id}_step7_storage_report.json")
    lm_edkg_output_path = args.output or (repo_root / f"{args.doc_id}_step8_lm_edkg.json")
    summary_output_path = args.summary_output or (repo_root / f"{args.doc_id}_step8_summary.json")

    for p in (concepts_path, triples_path, expansion_path, preprocessed_path, step7_report_path):
        if not p.exists():
            print(f"ERROR: required input file not found: {p}")
            return 1

    result = run_aggregation(
        doc_id=args.doc_id,
        concepts_path=str(concepts_path),
        triples_path=str(triples_path),
        expansion_path=str(expansion_path),
        preprocessed_path=str(preprocessed_path),
        step7_report_path=str(step7_report_path),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )

    # Save LM-EduKG
    with open(lm_edkg_output_path, "w", encoding="utf-8") as f:
        json.dump(result["lm_edkg"], f, indent=2)

    print(f"✓ Saved LM-EduKG: {lm_edkg_output_path}")

    # Save summary
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)

    print(f"✓ Saved summary: {summary_output_path}")

    # Print constraint results
    print("\n[Step 8] Constraint Verification:")
    for c in result["summary"]["constraints"]:
        status = "✓ PASS" if c["passed"] else "✗ FAIL"
        print(f"  {status} — {c['name']}")
        print(f"      {c['details']}")
        if c["warnings"]:
            print(f"      Warnings: {c['warnings'][:3]}")

    print(f"\n[Step 8] Results:")
    print(f"  All constraints passed: {result['summary']['all_constraints_passed']}")
    print(f"  Total concepts:    {result['summary']['total_concepts']}")
    print(f"  Total edges:       {result['summary']['total_edges']}")
    print(f"    - Extraction:    {result['summary']['extraction_edges']}")
    print(f"    - Expansion:     {result['summary']['expansion_edges']}")
    print(f"  SRS pool size:     {result['summary']['srs_pool_size']}")
    print(f"  Relation types:    {len(result['summary']['relation_distribution'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
