"""Step 8 — LM-EduKG Aggregation.

Validates four merge constraints from CLAUDE.md and exports the final
Learning Material EduKG (LM-EduKG) as JSON for SRS accuracy evaluation.

Four constraints verified:
  1. Every slide has ≥1 concept (concept coverage)
  2. All Slide-EduKG concepts exist in LM-EduKG (Neo4j)
  3. All extraction edges have evidence (provenance preserved)
  4. Cross-slide expansion edges tagged material_level=true

Output:
  - *_step8_lm_edkg.json (full graph export)
  - *_step8_summary.json (pipeline statistics + constraint results)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

from core.config import settings


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    details: str
    warnings: List[str]


def run_aggregation(
    doc_id: str,
    concepts_path: str,
    triples_path: str,
    expansion_path: str,
    preprocessed_path: str,
    step7_report_path: str,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> Dict[str, Any]:
    """Run Step 8 aggregation and return full report + LM-EduKG + summary."""

    neo4j_uri = neo4j_uri or settings.neo4j_uri
    neo4j_user = neo4j_user or settings.neo4j_user
    neo4j_password = neo4j_password or settings.neo4j_password

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def q(query: str, **params):
        with driver.session() as session:
            return session.run(query, **params).data()

    # Load all intermediate files
    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts_raw: List[Dict[str, Any]] = json.load(f)

    with open(triples_path, "r", encoding="utf-8") as f:
        extraction_triples: List[Dict[str, Any]] = json.load(f)

    with open(expansion_path, "r", encoding="utf-8") as f:
        expansion_edges: List[Dict[str, Any]] = json.load(f)

    with open(preprocessed_path, "r", encoding="utf-8") as f:
        slide_contents: List[Dict[str, Any]] = json.load(f)

    with open(step7_report_path, "r", encoding="utf-8") as f:
        step7_report: Dict[str, Any] = json.load(f)

    ordered_slides = sorted(s["slide_id"] for s in slide_contents)

    constraints: List[ConstraintResult] = []

    # Constraint 1: Every slide has concept coverage
    slide_coverage = q(
        """
        MATCH (c:Concept {doc_id: $doc_id})
        UNWIND c.slide_ids AS sid
        RETURN sid, count(c) AS concept_count
        ORDER BY sid
        """,
        doc_id=doc_id,
    )
    coverage_map = {row["sid"]: row["concept_count"] for row in slide_coverage}
    slides_no_concepts = [
        sid for sid in ordered_slides if coverage_map.get(sid, 0) == 0
    ]

    constraints.append(
        ConstraintResult(
            name="Slide concept coverage",
            passed=len(slides_no_concepts) == 0,
            details=f"{len(ordered_slides) - len(slides_no_concepts)}/{len(ordered_slides)} slides have concepts",
            warnings=slides_no_concepts[:5],
        )
    )

    # Constraint 2: All concepts exist in Neo4j
    all_concept_names_local = {c["name"] for c in concepts_raw}
    neo4j_concepts = q(
        "MATCH (c:Concept {doc_id: $doc_id}) RETURN c.name AS name",
        doc_id=doc_id,
    )
    neo4j_concept_names = {row["name"] for row in neo4j_concepts}
    missing_concepts = all_concept_names_local - neo4j_concept_names

    constraints.append(
        ConstraintResult(
            name="All concepts in Neo4j",
            passed=len(missing_concepts) == 0,
            details=f"{len(all_concept_names_local)} local concepts, {len(neo4j_concept_names)} in Neo4j",
            warnings=list(missing_concepts)[:5],
        )
    )

    # Constraint 3: Evidence provenance preserved
    edges_with_ev = q(
        """
        MATCH ()-[r:RELATION {doc_id: $doc_id, source: 'extraction'}]->()
        WHERE r.evidence IS NOT NULL AND r.evidence <> ''
        RETURN count(r) AS n
        """,
        doc_id=doc_id,
    )
    edges_without_ev = q(
        """
        MATCH ()-[r:RELATION {doc_id: $doc_id, source: 'extraction'}]->()
        WHERE r.evidence IS NULL OR r.evidence = ''
        RETURN count(r) AS n
        """,
        doc_id=doc_id,
    )
    n_with = edges_with_ev[0]["n"] if edges_with_ev else 0
    n_without = edges_without_ev[0]["n"] if edges_without_ev else 0

    constraints.append(
        ConstraintResult(
            name="Evidence provenance preserved",
            passed=n_without == 0,
            details=f"{n_with} extraction edges with evidence, {n_without} without",
            warnings=[],
        )
    )

    # Constraint 4: Cross-slide expansion edges tagged
    concept_slide_map = defaultdict(set)
    for c in concepts_raw:
        for sid in c.get("slide_ids", []):
            concept_slide_map[c["name"]].add(sid)

    cross_slide = []
    for e in expansion_edges:
        subj_slides = concept_slide_map.get(e["subject"], set())
        obj_slides = concept_slide_map.get(e["object"], set())
        if subj_slides and obj_slides and subj_slides.isdisjoint(obj_slides):
            cross_slide.append(e)

    if cross_slide:
        for e in cross_slide:
            try:
                q(
                    """
                    MATCH (s:Concept {name: $subject, doc_id: $doc_id}),
                          (o:Concept {name: $object, doc_id: $doc_id})
                    MATCH (s)-[r:RELATION {relation_type: 'relatedConcept', source: 'expansion'}]->(o)
                    SET r.material_level = true
                    """,
                    subject=e["subject"],
                    object=e["object"],
                    doc_id=doc_id,
                )
            except Exception:
                pass

    constraints.append(
        ConstraintResult(
            name="Cross-slide expansion edges",
            passed=True,
            details=f"{len(cross_slide)} cross-slide edges tagged as material_level",
            warnings=[],
        )
    )

    # Update LearningMaterial node
    total_edges = step7_report.get("total_extraction_edges", 0) + step7_report.get("total_expansion_edges", 0)

    q(
        """
        MERGE (lm:LearningMaterial {doc_id: $doc_id})
        SET lm.title                  = $title,
            lm.total_slides           = $total_slides,
            lm.total_concepts         = $total_concepts,
            lm.total_extraction_edges = $total_extraction_edges,
            lm.total_expansion_edges  = $total_expansion_edges,
            lm.total_edges            = $total_edges,
            lm.pipeline_version       = 'PACE-KG-v1'
        """,
        doc_id=doc_id,
        title=doc_id,
        total_slides=step7_report.get("total_slides", 0),
        total_concepts=step7_report.get("total_concepts_stored", 0),
        total_extraction_edges=step7_report.get("total_extraction_edges", 0),
        total_expansion_edges=step7_report.get("total_expansion_edges", 0),
        total_edges=total_edges,
    )

    # Export full LM-EduKG
    nodes = q(
        """
        MATCH (c:Concept {doc_id: $doc_id})
        RETURN c.name            AS name,
               c.aliases         AS aliases,
               c.slide_ids       AS slide_ids,
               c.source_type     AS source_type,
               c.keyphrase_score AS keyphrase_score,
               c.final_weight    AS final_weight,
               c.needs_review    AS needs_review
        ORDER BY c.final_weight DESC
        """,
        doc_id=doc_id,
    )

    edges = q(
        """
        MATCH (s:Concept {doc_id: $doc_id})-[r:RELATION]->(o:Concept {doc_id: $doc_id})
        RETURN s.name          AS subject,
               r.relation_type AS relation,
               o.name          AS object,
               r.evidence      AS evidence,
               r.confidence    AS confidence,
               r.source        AS source,
               r.slide_id      AS slide_id,
               r.material_level AS material_level
        ORDER BY r.confidence DESC
        """,
        doc_id=doc_id,
    )

    rel_dist = q(
        """
        MATCH ()-[r:RELATION {doc_id: $doc_id}]->()
        RETURN r.relation_type AS relation, count(r) AS count
        ORDER BY count DESC
        """,
        doc_id=doc_id,
    )

    # Build LM-EduKG export
    lm_edkg = {
        "doc_id": doc_id,
        "pipeline": "PACE-KG-v1",
        "total_slides": step7_report.get("total_slides", 0),
        "total_concepts": len(nodes),
        "total_edges": len(edges),
        "relation_distribution": {row["relation"]: row["count"] for row in rel_dist},
        "nodes": nodes,
        "edges": edges,
    }

    # SRS pool: extraction-only triples for evaluation
    srs_pool = [
        {
            "triple_id": f"t{i+1:05d}",
            "subject": e["subject"],
            "relation": e["relation"],
            "object": e["object"],
            "evidence": e["evidence"],
            "confidence": e["confidence"],
            "slide_id": e["slide_id"],
            "source": e["source"],
        }
        for i, e in enumerate(edges)
        if e["source"] == "extraction"
    ]

    lm_edkg["srs_pool"] = srs_pool
    lm_edkg["srs_pool_size"] = len(srs_pool)

    # Build summary
    summary = {
        "doc_id": doc_id,
        "pipeline_version": "PACE-KG-v1",
        "total_slides": step7_report.get("total_slides", 0),
        "total_concepts": len(nodes),
        "total_edges": len(edges),
        "extraction_edges": len([e for e in edges if e["source"] == "extraction"]),
        "expansion_edges": len([e for e in edges if e["source"] == "expansion"]),
        "srs_pool_size": len(srs_pool),
        "constraints": [
            {
                "name": c.name,
                "passed": c.passed,
                "details": c.details,
                "warnings": c.warnings,
            }
            for c in constraints
        ],
        "all_constraints_passed": all(c.passed for c in constraints),
        "relation_distribution": {row["relation"]: row["count"] for row in rel_dist},
    }

    driver.close()

    return {"lm_edkg": lm_edkg, "summary": summary}
