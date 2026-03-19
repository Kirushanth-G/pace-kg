"""Step 7 — Neo4j Slide-EduKG Storage.

Stores concepts + relations into Neo4j using the same strategy as in CLAUDE.md / pace_kg.py:
- Upsert all Concept nodes first
- Then upsert extraction edges and expansion edges slide-by-slide
- Create a LearningMaterial node and BELONGS_TO links

This module uses GraphDatabase from neo4j and leverages SBERT for semantic
conflict resolution (merging near-duplicate concepts).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase
from sentence_transformers import util as st_util

from api.models.triple import Triple, ExpansionEdge
from core.config import settings
from core.embeddings import get_sbert


MERGE_SIM_THRESHOLD = 0.92
REVIEW_SIM_THRESHOLD = 0.75


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_concept_name(name: str) -> str:
    return str(name).strip().lower()


def store_neo4j(
    doc_id: str,
    concepts_path: str,
    extraction_triples_path: str,
    expansion_edges_path: str,
    preprocessed_path: str,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> Dict[str, Any]:
    """Store the full Slide-EduKG in Neo4j.

    Returns a report dict describing what was written.
    """

    # Load inputs
    concepts_raw: List[Dict[str, Any]] = _load_json(concepts_path)
    extraction_triples: List[Dict[str, Any]] = _load_json(extraction_triples_path)
    expansion_edges: List[Dict[str, Any]] = _load_json(expansion_edges_path)
    slide_contents: List[Dict[str, Any]] = _load_json(preprocessed_path)

    # Prep lookups
    slide_text_map = {s["slide_id"]: s["clean_text"] for s in slide_contents}
    ordered_slides = sorted(slide_text_map.keys())

    concept_by_name: Dict[str, Dict[str, Any]] = {}
    for c in concepts_raw:
        name = _normalize_concept_name(c.get("name", ""))
        if not name:
            continue
        concept_by_name[name] = c
        for alias in c.get("aliases", []):
            concept_by_name[_normalize_concept_name(alias)] = c

    # Pre-encode concept names for semantic merge
    sbert = get_sbert()
    concept_embs = {
        name: sbert.encode(name, convert_to_tensor=True, show_progress_bar=False)
        for name in list(concept_by_name.keys())
    }

    def resolve_concept_name(name: str) -> str:
        name_l = _normalize_concept_name(name)
        if name_l in concept_by_name:
            return concept_by_name[name_l]["name"]

        emb = sbert.encode(name_l, convert_to_tensor=True, show_progress_bar=False)
        for cname, cemb in concept_embs.items():
            sim = float(st_util.cos_sim(emb, cemb))
            if sim >= MERGE_SIM_THRESHOLD:
                return concept_by_name[cname]["name"]
        return name_l

    neo4j_uri = neo4j_uri or settings.neo4j_uri
    neo4j_user = neo4j_user or settings.neo4j_user
    neo4j_password = neo4j_password or settings.neo4j_password

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def _run_query(tx, query, **params):
        return tx.run(query, **params)

    # Ensure constraints/indexes exist
    with driver.session() as session:
        session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        session.run("CREATE INDEX concept_doc IF NOT EXISTS FOR (c:Concept) ON (c.doc_id)")
        session.run("CREATE INDEX lm_doc IF NOT EXISTS FOR (l:LearningMaterial) ON (l.doc_id)")

    # Cypher templates
    UPSERT_CONCEPT = """
MERGE (c:Concept {name: $name})
SET   c.aliases         = $aliases,
      c.slide_ids       = $slide_ids,
      c.source_type     = $source_type,
      c.keyphrase_score = $keyphrase_score,
      c.final_weight    = $final_weight,
      c.doc_id          = $doc_id,
      c.needs_review    = $needs_review
RETURN c.name AS name
"""

    CREATE_EXTRACTION_EDGE = """
MATCH (s:Concept {name: $subject}), (o:Concept {name: $object})
MERGE (s)-[r:RELATION {relation_type: $relation_type, slide_id: $slide_id, doc_id: $doc_id}]->(o)
SET r.evidence   = $evidence,
    r.confidence = $confidence,
    r.source     = 'extraction'
RETURN type(r) AS rel
"""

    CREATE_EXPANSION_EDGE = """
MATCH (s:Concept {name: $subject}), (o:Concept {name: $object})
MERGE (s)-[r:RELATION {relation_type: 'relatedConcept', slide_id: $slide_id, doc_id: $doc_id}]->(o)
SET r.confidence = $confidence,
    r.source     = 'expansion'
RETURN type(r) AS rel
"""

    UPSERT_LM = """
MERGE (lm:LearningMaterial {doc_id: $doc_id})
SET   lm.title        = $title,
      lm.total_slides = $total_slides
RETURN lm.doc_id AS doc_id
"""

    LINK_CONCEPT_TO_LM = """
MATCH (c:Concept {name: $name, doc_id: $doc_id}),
      (lm:LearningMaterial {doc_id: $doc_id})
MERGE (c)-[r:BELONGS_TO]->(lm)
SET r.final_weight = $final_weight
"""

    report: Dict[str, Any] = {
        "doc_id": doc_id,
        "total_slides": len(ordered_slides),
        "slides_stored": [],
        "total_concepts_stored": 0,
        "total_extraction_edges": 0,
        "total_expansion_edges": 0,
        "errors": [],
    }

    # Pass 1: Upsert all concept nodes
    with driver.session() as session:
        stored = 0
        for c in concepts_raw:
            name = _normalize_concept_name(c.get("name", ""))
            if not name:
                continue
            try:
                session.execute_write(
                    _run_query,
                    UPSERT_CONCEPT,
                    name=name,
                    aliases=c.get("aliases", []),
                    slide_ids=c.get("slide_ids", []),
                    source_type=c.get("source_type", "body"),
                    keyphrase_score=float(c.get("keyphrase_score", 0.0)),
                    final_weight=float(c.get("final_weight", 0.0)),
                    doc_id=doc_id,
                    needs_review=bool(c.get("needs_review", False)),
                )
                stored += 1
            except Exception as e:
                report["errors"].append(f"Concept '{name}': {e}")

        report["total_concepts_stored"] = stored

    # Pass 2: Store edges slide-by-slide
    triples_by_slide = defaultdict(list)
    for t in extraction_triples:
        triples_by_slide[str(t.get("slide_id", ""))].append(t)

    expansion_by_slide = defaultdict(list)
    for e in expansion_edges:
        expansion_by_slide[str(e.get("slide_id", ""))].append(e)

    for slide_id in ordered_slides:
        slide_report = {"slide_id": slide_id, "extraction_edges": 0, "expansion_edges": 0}
        with driver.session() as session:
            for t in triples_by_slide.get(slide_id, []):
                subj = resolve_concept_name(t.get("subject", ""))
                obj = resolve_concept_name(t.get("object", ""))
                try:
                    session.execute_write(
                        _run_query,
                        CREATE_EXTRACTION_EDGE,
                        subject=subj,
                        object=obj,
                        relation_type=t.get("relation", ""),
                        evidence=t.get("evidence", ""),
                        confidence=float(t.get("confidence", 0.0)),
                        slide_id=slide_id,
                        doc_id=doc_id,
                    )
                    slide_report["extraction_edges"] += 1
                except Exception as e:
                    report["errors"].append(f"Extraction edge {subj}->{obj}: {e}")

            for e in expansion_by_slide.get(slide_id, []):
                subj = resolve_concept_name(e.get("subject", ""))
                obj = resolve_concept_name(e.get("object", ""))
                try:
                    session.execute_write(
                        _run_query,
                        CREATE_EXPANSION_EDGE,
                        subject=subj,
                        object=obj,
                        confidence=float(e.get("confidence", 0.0)),
                        slide_id=slide_id,
                        doc_id=doc_id,
                    )
                    slide_report["expansion_edges"] += 1
                except Exception as e:
                    report["errors"].append(f"Expansion edge {subj}->{obj}: {e}")

        report["slides_stored"].append(slide_report)
        report["total_extraction_edges"] += slide_report["extraction_edges"]
        report["total_expansion_edges"] += slide_report["expansion_edges"]

    # Pass 3: LearningMaterial node + BELONGS_TO links
    with driver.session() as session:
        session.execute_write(
            _run_query,
            UPSERT_LM,
            doc_id=doc_id,
            title=f"{doc_id}",
            total_slides=len(ordered_slides),
        )

        for c in concepts_raw:
            name = _normalize_concept_name(c.get("name", ""))
            if not name:
                continue
            try:
                session.execute_write(
                    _run_query,
                    LINK_CONCEPT_TO_LM,
                    name=name,
                    doc_id=doc_id,
                    final_weight=float(c.get("final_weight", 0.0)),
                )
            except Exception as e:
                report["errors"].append(f"BELONGS_TO {name}: {e}")

    driver.close()
    return report
