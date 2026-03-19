"""Step 5 — Concept Weighting & Pruning.

Implements the logic described in CLAUDE.md / pace_kg.py for computing a
final importance weight for each concept, merging near-duplicates, and pruning
low-weight concepts and any triples that reference them.

The output is a list of ConceptNode objects and a pruned triple list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

from sentence_transformers import util as st_util

from api.models.concept import Keyphrase
from api.models.triple import Triple
from core.config import settings
from core.embeddings import get_sbert
from pipeline.step2_preprocessor.cleaner import SlideContent


# -- Constants (mirrors CLAUDE.md / pace_kg.py) ------------------------------
RELATION_ROLE_BOOSTS: Dict[Tuple[str, str], float] = {
    ("isDefinedAs",        "object"):  +0.15,
    ("isPrerequisiteOf",   "subject"): +0.10,
    ("isGeneralizationOf", "object"):  +0.10,
    ("contrastedWith",     "subject"): +0.05,
    ("contrastedWith",     "object"):  +0.05,
    ("causeOf",            "subject"): +0.05,
    ("isExampleOf",        "subject"): -0.05,
}

SOURCE_TYPE_BOOSTS: Dict[str, float] = {
    "heading": +0.10,
    "bullet":  +0.05,
    "table":   +0.05,
    "body":    +0.00,
    "caption": -0.05,
}

MERGE_SIM_THRESHOLD = 0.92
REVIEW_SIM_THRESHOLD = 0.75


@dataclass
class ConceptNode:
    name: str
    aliases: List[str]
    slide_ids: List[str]
    source_type: str
    keyphrase_score: float
    final_weight: float
    doc_id: str
    needs_review: bool = False


def load_keyphrases(path: str) -> Dict[str, List[Keyphrase]]:
    """Load keyphrases JSON into slide_id -> List[Keyphrase]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[Keyphrase]] = {}
    for slide_id, items in data.items():
        if not isinstance(items, list):
            continue
        kps: List[Keyphrase] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            phrase = item.get("phrase")
            if not isinstance(phrase, str):
                continue
            kps.append(Keyphrase(
                phrase=phrase.strip().lower(),
                score=float(item.get("score", 0.0)),
                source_type=str(item.get("source_type", "body")),
                slide_id=slide_id,
                doc_id=str(item.get("doc_id", "")),
                appears_in=str(item.get("appears_in", "")),
            ))
        out[slide_id] = kps
    return out


def load_triples(path: str) -> List[Triple]:
    """Load triples JSON into a list of Triple dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples: List[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        triples.append(Triple(
            subject=str(item.get("subject", "")).strip().lower(),
            relation=str(item.get("relation", "")).strip(),
            object=str(item.get("object", "")).strip().lower(),
            evidence=str(item.get("evidence", "")),
            confidence=float(item.get("confidence", 0.0)),
            slide_id=str(item.get("slide_id", "")),
            doc_id=str(item.get("doc_id", "")),
            source=str(item.get("source", "extraction")),
        ))
    return triples


def run_weighting(
    slides: List[SlideContent],
    keyphrases_by_slide: Dict[str, List[Keyphrase]],
    triples: List[Triple],
    doc_id: str,
    weight_threshold: Optional[float] = None,
    merge_threshold: Optional[float] = None,
    review_threshold: Optional[float] = None,
) -> tuple[List[ConceptNode], List[Triple]]:
    """Compute final concept weights and prune low-weight concepts.

    Returns:
        (concept_nodes, pruned_triples)
    """

    weight_threshold = (
        settings.weight_pruning_threshold
        if weight_threshold is None
        else weight_threshold
    )
    merge_threshold = MERGE_SIM_THRESHOLD if merge_threshold is None else merge_threshold
    review_threshold = REVIEW_SIM_THRESHOLD if review_threshold is None else review_threshold

    sbert = get_sbert()

    # Build lookup structures
    slide_text_map: Dict[str, str] = {s.slide_id: s.clean_text for s in slides}
    full_doc_text = " ".join(s.clean_text for s in slides if s.clean_text.strip())
    doc_emb = sbert.encode(full_doc_text, convert_to_tensor=True, show_progress_bar=False)

    # Concept source map (from keyphrases + triples)
    concept_source_map: Dict[str, dict] = {}

    for slide_id, kps in keyphrases_by_slide.items():
        for kp in kps:
            phrase = kp.phrase.strip().lower()
            if phrase not in concept_source_map:
                concept_source_map[phrase] = {
                    "source_type":     kp.source_type,
                    "keyphrase_score": kp.score,
                    "slide_ids":       [],
                }
            if slide_id not in concept_source_map[phrase]["slide_ids"]:
                concept_source_map[phrase]["slide_ids"].append(slide_id)

    for t in triples:
        for role in ("subject", "object"):
            phrase = getattr(t, role).strip().lower()
            if phrase not in concept_source_map:
                concept_source_map[phrase] = {
                    "source_type":     "body",
                    "keyphrase_score": 0.0,
                    "slide_ids":       [],
                }
            if t.slide_id not in concept_source_map[phrase]["slide_ids"]:
                concept_source_map[phrase]["slide_ids"].append(t.slide_id)

    # Role boosts
    concept_role_boost: Dict[str, float] = {p: 0.0 for p in concept_source_map}
    for t in triples:
        subj = t.subject
        obj = t.object
        rel = t.relation
        concept_role_boost[subj] = concept_role_boost.get(subj, 0.0) + RELATION_ROLE_BOOSTS.get((rel, "subject"), 0.0)
        concept_role_boost[obj]  = concept_role_boost.get(obj,  0.0) + RELATION_ROLE_BOOSTS.get((rel, "object"),  0.0)

    # Evidence map
    concept_evidence_map: Dict[str, List[str]] = {p: [] for p in concept_source_map}
    for t in triples:
        concept_evidence_map[t.subject].append(t.evidence)
        concept_evidence_map[t.object].append(t.evidence)

    # Compute weights
    def _compute_weight(phrase: str) -> float:
        info = concept_source_map[phrase]
        slide_ids = info["slide_ids"]
        src_type = info["source_type"]

        phrase_emb = sbert.encode(phrase, convert_to_tensor=True, show_progress_bar=False)

        # w_evidence
        evidence_sents = concept_evidence_map.get(phrase, [])
        if evidence_sents:
            best_ev_sim = max(
                float(st_util.cos_sim(
                    phrase_emb,
                    sbert.encode(ev, convert_to_tensor=True, show_progress_bar=False),
                ))
                for ev in evidence_sents
            )
        else:
            best_ev_sim = 0.0

        # w_slide
        if slide_ids:
            slide_texts = [slide_text_map.get(sid, "") for sid in slide_ids if slide_text_map.get(sid)]
            if slide_texts:
                combined_slide = " ".join(slide_texts)
                slide_emb = sbert.encode(combined_slide, convert_to_tensor=True, show_progress_bar=False)
                w_slide = float(st_util.cos_sim(phrase_emb, slide_emb))
            else:
                w_slide = 0.0
        else:
            w_slide = 0.0

        # w_doc
        w_doc = float(st_util.cos_sim(phrase_emb, doc_emb))

        raw = (0.5 * best_ev_sim) + (0.3 * w_slide) + (0.2 * w_doc)
        role_boost = min(concept_role_boost.get(phrase, 0.0), 0.20)
        src_boost = SOURCE_TYPE_BOOSTS.get(src_type, 0.0)
        final = min(raw + role_boost + src_boost, 1.0)
        return round(final, 4)

    concept_weights: Dict[str, float] = {}
    for phrase in concept_source_map:
        concept_weights[phrase] = _compute_weight(phrase)

    # Merge near-duplicates & flag review
    phrases_sorted = sorted(concept_weights.keys(), key=lambda p: -concept_weights[p])
    merged_into: Dict[str, str] = {}
    needs_review: Set[str] = set()

    # Precompute embeddings for merge loop
    phrase_embs = {
        p: sbert.encode(p, convert_to_tensor=True, show_progress_bar=False)
        for p in phrases_sorted
    }

    for i, phrase in enumerate(phrases_sorted):
        if phrase in merged_into:
            continue
        for other in phrases_sorted[i+1:]:
            if other in merged_into:
                continue
            sim = float(st_util.cos_sim(phrase_embs[phrase], phrase_embs[other]))
            if sim >= merge_threshold:
                merged_into[other] = phrase
            elif sim >= review_threshold:
                needs_review.add(phrase)
                needs_review.add(other)

    # Build ConceptNodes
    concept_nodes: Dict[str, ConceptNode] = {}

    for phrase, info in concept_source_map.items():
        canonical = merged_into.get(phrase, phrase)
        if canonical != phrase:
            continue

        aliases = [other for other, canon in merged_into.items() if canon == phrase]

        # Merge slide_ids from aliases
        all_slide_ids = list(info["slide_ids"])
        for alias in aliases:
            for sid in concept_source_map.get(alias, {}).get("slide_ids", []):
                if sid not in all_slide_ids:
                    all_slide_ids.append(sid)

        concept_nodes[phrase] = ConceptNode(
            name=phrase,
            aliases=aliases,
            slide_ids=sorted(all_slide_ids),
            source_type=info["source_type"],
            keyphrase_score=info["keyphrase_score"],
            final_weight=concept_weights[phrase],
            doc_id=doc_id,
            needs_review=phrase in needs_review,
        )

    # Prune by weight
    pruned_names: Set[str] = {
        name for name, node in concept_nodes.items() if node.final_weight < weight_threshold
    }
    for name in pruned_names:
        concept_nodes.pop(name, None)

    surviving_names: Set[str] = set(concept_nodes.keys())
    for node in concept_nodes.values():
        surviving_names.update(node.aliases)

    kept_triples: List[Triple] = []
    for t in triples:
        if t.subject in surviving_names and t.object in surviving_names:
            kept_triples.append(t)

    return list(concept_nodes.values()), kept_triples


def save_concepts(concepts: List[ConceptNode], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in concepts], f, ensure_ascii=False, indent=2)


def save_triples(triples: List[Triple], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in triples], f, ensure_ascii=False, indent=2)
