"""Step 6 — Closed-Corpus Concept Expansion.

Implements the closed-corpus expansion pipeline described in CLAUDE.md / pace_kg.py:
- Builds a document vocabulary from keyphrases + extraction triples + noun chunks
- Uses Groq LLM to select related concepts from a candidate pool
- Applies SBERT gating and slide-scope constraints
- Outputs a list of ExpansionEdge objects (relatedConcept edges)

This module is intentionally self-contained and uses the same thresholds and
behaviors as the Colab pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Dict, List, Set

from sentence_transformers import util as st_util
from langchain_core.messages import HumanMessage, SystemMessage

from api.models.concept import Keyphrase
from api.models.triple import ExpansionEdge, Triple
from core.config import settings
from core.embeddings import get_sbert
from core.llm_client import get_fallback_llm, get_llm
from pipeline.step2_preprocessor.cleaner import SlideContent


# Constants (from CLAUDE.md / pace_kg.py)
SBERT_SIM_THRESHOLD = settings.expansion_similarity_threshold
DOC_WEIGHT_THRESHOLD = 0.70
MAX_CANDIDATES_PER_CONCEPT = settings.expansion_max_related


def _invoke_with_fallback(messages: list, max_retries: int = 3) -> str | None:
    for attempt in range(1, max_retries + 1):
        try:
            return str(get_llm().invoke(messages).content)
        except Exception as e:
            text = str(e).lower()
            if "429" in text or "rate" in text:
                try:
                    return str(get_fallback_llm().invoke(messages).content)
                except Exception as e2:
                    text2 = str(e2).lower()
                    if "429" in text2 or "rate" in text2:
                        # backoff
                        continue
                    raise
            raise
    return None


SYSTEM_PROMPT = (
    "You are a knowledge graph assistant for educational materials.\n"
    "Given a main concept and its slide context, select the most educationally "
    "relevant related concepts from the candidate pool.\n\n"
    "STRICT RULES:\n"
    "1. Select ONLY concepts from the candidate pool. Do not invent anything.\n"
    "2. Choose concepts meaningfully related in an educational sense.\n"
    f"3. Select at most {MAX_CANDIDATES_PER_CONCEPT} concepts.\n"
    '4. Return ONLY valid JSON: {"related": ["concept1", "concept2"]}\n'
    "5. If nothing is relevant return: {\"related\": []}\n"
    "6. Do not explain your choices."
)

USER_PROMPT_TEMPLATE = (
    "Main concept: '{concept}'\n"
    "Slide context: '{slide_context}'\n\n"
    "Candidate pool (select ONLY from this list):\n"
    "{pool}\n\n"
    'Return JSON with key "related" containing an array of selected strings.'
)


def _parse_llm_response(raw: str) -> List[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "related" in parsed and isinstance(parsed["related"], list):
            return [str(x).strip().lower() for x in parsed["related"] if isinstance(x, str)]
    except Exception:
        pass
    return []


def run_expansion(
    slides: List[SlideContent],
    keyphrases_by_slide: Dict[str, List[str]],
    pruned_triples: List[Triple],
    doc_id: str,
    max_candidates: int | None = None,
    sbert_threshold: float | None = None,
    doc_weight_threshold: float | None = None,
) -> List[ExpansionEdge]:
    """Generate closed-corpus expansion edges for the document."""

    max_candidates = MAX_CANDIDATES_PER_CONCEPT if max_candidates is None else max_candidates
    sbert_threshold = SBERT_SIM_THRESHOLD if sbert_threshold is None else sbert_threshold
    doc_weight_threshold = DOC_WEIGHT_THRESHOLD if doc_weight_threshold is None else doc_weight_threshold

    sbert = get_sbert()

    # Build slide lookup structures
    slide_text_map: Dict[str, str] = {s.slide_id: s.clean_text for s in slides}
    ordered_slides = sorted(slide_text_map.keys())
    slide_index: Dict[str, int] = {sid: i for i, sid in enumerate(ordered_slides)}

    full_doc_text = " ".join(s.clean_text for s in slides if s.clean_text.strip())
    doc_emb = sbert.encode(full_doc_text, convert_to_tensor=True, show_progress_bar=False)

    # Build vocabulary (keyphrases + triple concepts + noun chunks)
    concept_slide_map: Dict[str, Set[str]] = {}

    for slide_id, kps in keyphrases_by_slide.items():
        for kp in kps:
            if isinstance(kp, str):
                phrase = kp
            elif isinstance(kp, dict):
                phrase = kp.get("phrase", "")
            elif isinstance(kp, Keyphrase):
                phrase = kp.phrase
            else:
                continue
            phrase = phrase.strip().lower()
            if not phrase:
                continue
            concept_slide_map.setdefault(phrase, set()).add(slide_id)

    # Add concepts from pruned triples (subject/object) into vocabulary
    for t in pruned_triples:
        if isinstance(t, dict):
            subj = str(t.get("subject", "")).strip().lower()
            obj = str(t.get("object", "")).strip().lower()
            slide_id = str(t.get("slide_id", ""))
        elif isinstance(t, Triple):
            subj = t.subject.strip().lower()
            obj = t.object.strip().lower()
            slide_id = t.slide_id
        else:
            continue
        for phrase in (subj, obj):
            if phrase:
                concept_slide_map.setdefault(phrase, set()).add(slide_id)

    # Add noun chunks (closed-corpus expansion), but avoid duplicates if stripped article already exists
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        for s in slides:
            if not s.clean_text.strip():
                continue
            doc = nlp(s.clean_text)
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                stripped = re.sub(r"^(a|an|the)\s+", "", phrase).strip()
                if stripped in concept_slide_map:
                    continue
                if len(phrase) >= 3 and not all(tok.is_stop for tok in chunk):
                    concept_slide_map.setdefault(phrase, set()).add(s.slide_id)
    except Exception:
        # If spaCy not available, skip noun chunk expansion.
        pass

    doc_vocabulary = sorted(concept_slide_map.keys())

    # Normalize triples (allow dicts from JSON or Triple objects)
    normalized_triples: List[Triple] = []
    for t in pruned_triples:
        if isinstance(t, dict):
            normalized_triples.append(Triple(
                subject=str(t.get("subject", "")).strip().lower(),
                relation=str(t.get("relation", "")).strip(),
                object=str(t.get("object", "")).strip().lower(),
                evidence=str(t.get("evidence", "")),
                confidence=float(t.get("confidence", 0.0)) if t.get("confidence") is not None else 0.0,
                slide_id=str(t.get("slide_id", "")),
                doc_id=str(t.get("doc_id", "")),
                source=str(t.get("source", "extraction")),
            ))
        elif isinstance(t, Triple):
            normalized_triples.append(t)

    # Existing pairs (avoid duplicates)
    existing_pairs = set()
    for t in normalized_triples:
        s = t.subject.lower().strip()
        o = t.object.lower().strip()
        existing_pairs.add((s, o))
        existing_pairs.add((o, s))

    # Pre-encode all vocabulary items
    vocab_embs = {
        p: sbert.encode(p, convert_to_tensor=True, show_progress_bar=False)
        for p in doc_vocabulary
    }

    core_concepts = sorted({
        getattr(t, r).lower().strip()
        for t in normalized_triples
        for r in ("subject", "object")
    })

    expansion_edges: List[ExpansionEdge] = []

    for concept in core_concepts:
        concept_slides = sorted(concept_slide_map.get(concept, []))
        if not concept_slides:
            continue

        primary_slide = concept_slides[0]
        slide_context = slide_text_map.get(primary_slide, "")[:400]

        already_linked = {
            o for (s, o) in existing_pairs if s == concept
        } | {
            s for (s, o) in existing_pairs if o == concept
        }

        pool = [
            p for p in doc_vocabulary
            if p != concept and p not in already_linked and len(p) >= 3
        ]
        if not pool:
            continue

        pool_str = "\n".join(f"- {p}" for p in pool[:80])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(
                concept=concept,
                slide_context=slide_context,
                pool=pool_str,
            )),
        ]

        raw = _invoke_with_fallback(messages)
        candidates = _parse_llm_response(raw or "")

        candidates = [c for c in candidates if c in vocab_embs]
        if not candidates:
            continue

        if concept in vocab_embs:
            concept_emb = vocab_embs[concept]
        else:
            concept_emb = sbert.encode(concept, convert_to_tensor=True, show_progress_bar=False)

        # SBERT gate
        sbert_passed = []
        for cand in candidates:
            sim = float(st_util.cos_sim(concept_emb, vocab_embs[cand]))
            if sim >= sbert_threshold:
                sbert_passed.append((cand, round(sim, 4)))

        # Slide-scope constraint
        concept_indices = {slide_index[s] for s in concept_slides if s in slide_index}
        for cand, sim in sbert_passed:
            cand_slides = concept_slide_map.get(cand, set())
            cand_indices = {slide_index[s] for s in cand_slides if s in slide_index}
            adjacent = any(abs(ci - cj) <= 1 for ci in concept_indices for cj in cand_indices)
            doc_sim = float(st_util.cos_sim(vocab_embs[cand], doc_emb))
            high_doc_rel = doc_sim >= doc_weight_threshold

            if not (adjacent or high_doc_rel):
                continue

            # Skip self-loop and trivial article variants
            if cand == concept:
                continue
            stripped_cand = re.sub(r"^(a|an|the)\s+", "", cand).strip()
            if stripped_cand == concept:
                continue

            pair = (concept, cand)
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            existing_pairs.add((cand, concept))

            expansion_edges.append(ExpansionEdge(
                subject=concept,
                relation="relatedConcept",
                object=cand,
                source="expansion",
                confidence=sim,
                slide_id=primary_slide,
                doc_id=doc_id,
            ))

    return expansion_edges


def save_expansion_edges(edges: List[ExpansionEdge], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in edges], f, ensure_ascii=False, indent=2)
