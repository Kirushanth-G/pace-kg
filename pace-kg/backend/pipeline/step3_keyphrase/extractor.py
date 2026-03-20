"""Step 3 — Keyphrase Extraction (GLiNER + SBERT).

This module mirrors the Colab pipeline (CLAUDE.md Step 3):
- Uses GLiNER large-v2.1 for zero-shot entity/keyphrase extraction.
- Uses SBERT (all-MiniLM-L6-v2) for de-duplication (semantic overlap).

Note: GLiNER must be installed (added to requirements.txt) so this module
works the same as the Colab/CLAUDE version.

The goal is to extract educational concepts (keyphrases) from each slide without
relying on external knowledge bases (closed-corpus).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import gliner
import spacy
from sentence_transformers import util

from api.models.concept import Keyphrase
from core.config import Settings
from core.embeddings import get_minilm
from pipeline.step2_preprocessor.cleaner import SlideContent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER configuration (matches CLAUDE.md)
# ---------------------------------------------------------------------------
GLINER_MODEL = "urchade/gliner_large-v2.1"
GLINER_LABELS = [
    "Academic Concept",
    "Theoretical Principle",
    "Technical Term",
    "Process or Method",
    "System or Framework",
    "Formula or Equation",
]
GLINER_THRESHOLD = 0.35
KEYPHRASE_MAX = 25
DEDUP_SIM_THRESHOLD = 0.85
HEADING_BOOST = 0.15

_gliner: gliner.GLiNER | None = None
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_gliner() -> gliner.GLiNER:
    global _gliner
    if _gliner is None:
        _gliner = gliner.GLiNER.from_pretrained(GLINER_MODEL)
    return _gliner


def _normalize_phrase(phrase: str) -> str:
    return str(phrase).strip().lower()


def _build_extract_text(slide: SlideContent) -> str:
    """Build the text passed to GLiNER (headings first, code excluded).

    Mirrors the Colab pipeline exactly:
      heading_text + ". " + rest_text
    so GLiNER sees headings with higher positional weight, and no content
    is duplicated (clean_text is NOT appended here).
    """
    heading_text = " ".join(slide.headings)
    rest_text = " ".join(
        slide.body_text + slide.bullets + slide.table_cells + slide.captions
    )
    if heading_text and rest_text:
        return heading_text + ". " + rest_text
    return heading_text or rest_text


def _extract_gliner_candidates(extract_text: str) -> Dict[str, float]:
    """Extract candidate keyphrases using GLiNER and return best score per phrase."""
    if not extract_text.strip():
        return {}

    model = _get_gliner()
    entities = model.predict_entities(
        extract_text,
        labels=GLINER_LABELS,
        threshold=GLINER_THRESHOLD,
        flat_ner=True,
    )

    best_scores: Dict[str, float] = {}
    for ent in entities:
        phrase = _normalize_phrase(ent.get("text", ""))
        if not phrase or len(phrase) < 3:
            continue
        score = float(ent.get("score", 0.0))
        if score <= 0:
            continue
        # Keep highest score per phrase
        if phrase not in best_scores or score > best_scores[phrase]:
            best_scores[phrase] = score

    return best_scores


def _assign_source_type(phrase: str, slide: SlideContent) -> str:
    """Assign source bucket — check headings first, then bullets, table, caption, body."""
    pl = phrase.lower()
    for h in slide.headings:
        if pl in h.lower():
            return "heading"
    for b in slide.bullets:
        if pl in b.lower():
            return "bullet"
    for t in slide.table_cells:
        if pl in t.lower():
            return "table"
    for c in slide.captions:
        if pl in c.lower():
            return "caption"
    return "body"


def _find_sentence(phrase: str, clean_text: str) -> str:
    """Return the sentence from clean_text that contains the phrase.

    Uses spaCy sentence segmentation. Falls back to first 200 chars.
    """
    nlp = _get_nlp()
    doc = nlp(clean_text)
    pl = phrase.lower()
    for sent in doc.sents:
        if pl in sent.text.lower():
            return sent.text.strip()
    return clean_text[:200]


def _dedupe_phrases(phrases: Dict[str, float]) -> List[tuple[str, float]]:
    """Deduplicate near-synonyms using SBERT cosine similarity."""
    model = get_minilm()
    sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)

    kept: List[tuple[str, float]] = []
    kept_embs: List[Any] = []

    for phrase, score in sorted_phrases:
        emb = model.encode(phrase, convert_to_tensor=True, show_progress_bar=False)
        duplicate = False
        for ke in kept_embs:
            if float(util.cos_sim(emb, ke)) >= DEDUP_SIM_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((phrase, score))
        kept_embs.append(emb)
        if len(kept) >= KEYPHRASE_MAX:
            break

    return kept


def extract_keyphrases(slide: SlideContent, config: Settings) -> List[Keyphrase]:
    """Extract keyphrases from a single slide using GLiNER + SBERT."""
    if not slide.clean_text.strip():
        logger.debug("Skipping empty slide %s", slide.slide_id)
        return []

    # Skip title slides (page 1) or very short slides
    if slide.page_number == 1 and len(slide.clean_text.split()) < 40:
        return []
    if len(slide.clean_text.split()) < 8:
        return []

    extract_text = _build_extract_text(slide)
    scores = _extract_gliner_candidates(extract_text)
    if not scores:
        return []

    deduped = _dedupe_phrases(scores)

    keyphrases: list[Keyphrase] = []
    for phrase, score in deduped:
        source_type = _assign_source_type(phrase, slide)
        final_score = min(score + HEADING_BOOST, 1.0) if source_type == "heading" else score
        appears_in = _find_sentence(phrase, slide.clean_text)

        keyphrases.append(
            Keyphrase(
                phrase=phrase,
                score=round(final_score, 4),
                source_type=source_type,
                slide_id=slide.slide_id,
                doc_id=slide.doc_id,
                appears_in=appears_in,
            )
        )

    return keyphrases


def extract_keyphrases_all(slides: List[SlideContent], config: Settings) -> dict[str, List[Keyphrase]]:
    """Extract keyphrases for all slides. Returns {slide_id: [Keyphrase, ...]}."""
    results: dict[str, List[Keyphrase]] = {}
    for slide in slides:
        results[slide.slide_id] = extract_keyphrases(slide, config)
    return results