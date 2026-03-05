"""Step 3 — Keyphrase Extraction (SIFRank-style with sentence-transformers).

Implements the SIFRank algorithm spirit using:
  - spaCy en_core_web_sm  : candidate extraction (noun chunks) + linguistic validation
  - all-MiniLM-L6-v2      : phrase and document embeddings (80MB — 8GB friendly)

Algorithm (mirrors SIFRankSqueezeBERT from coursemapper-kg):
  1. Extract candidate noun chunks from clean_text
  2. Score by cosine similarity to document embedding  (SIFRank-style)
  3. Apply CLAUDE.md adaptive filter pipeline (quality, linguistic, noun-chunk, source, heading boost)
  4. Return List[Keyphrase] sorted by final score descending

Reference implementation:
  services/sifrank/KeyphraseService.extract_keyphrases()
  → replaced StanfordCoreNLPTagger with spaCy (no Java / large JARs needed)
  → replaced Flair TransformerWordEmbeddings with sentence-transformers MiniLM
"""
from __future__ import annotations

import logging
import re
from typing import List

import spacy
from sentence_transformers import util

from api.models.concept import Keyphrase
from core.config import Settings
from core.embeddings import get_minilm
from pipeline.step2_preprocessor.cleaner import SlideContent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy singleton
# ---------------------------------------------------------------------------
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------
def _extract_candidates(clean_text: str) -> list[str]:
    """Extract unique lowercase noun-chunk candidates from text via spaCy."""
    nlp = _get_nlp()
    doc = nlp(clean_text)
    seen: set[str] = set()
    candidates: list[str] = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # Remove leading determiners like "the", "a", "an"
        phrase = re.sub(r"^(the|a|an)\s+", "", phrase).strip()
        if len(phrase) >= 3 and phrase not in seen:
            seen.add(phrase)
            candidates.append(phrase)
    return candidates


# ---------------------------------------------------------------------------
# SIFRank-style scoring
# ---------------------------------------------------------------------------
def _sifrank_scores(candidates: list[str], doc_text: str) -> dict[str, float]:
    """Score each candidate by cosine similarity to the document embedding.

    Batches all encodings in one call for efficiency.
    Mirrors the SIFSentenceEmbeddings + cos_sim_distance logic.
    """
    if not candidates or not doc_text.strip():
        return {}

    model = get_minilm()
    all_texts = [doc_text] + candidates
    embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)

    doc_emb = embeddings[0]
    cand_embs = embeddings[1:]

    scores: dict[str, float] = {}
    for phrase, cand_emb in zip(candidates, cand_embs):
        scores[phrase] = float(util.cos_sim(doc_emb.unsqueeze(0), cand_emb.unsqueeze(0)))

    return scores


# ---------------------------------------------------------------------------
# Adaptive filter — Step 3 from CLAUDE.md
# ---------------------------------------------------------------------------
def _is_valid(phrase: str) -> bool:
    """spaCy linguistic filter: must contain a noun, not be all-stop, len >= 3."""
    nlp = _get_nlp()
    doc = nlp(phrase)
    has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in doc)
    all_stop = all(t.is_stop for t in doc)
    return has_noun and not all_stop and len(phrase.strip()) >= 3


def _in_noun_chunks(phrase: str, clean_text: str) -> bool:
    """Noun-chunk cross-validation: phrase must appear as a spaCy noun chunk."""
    nlp = _get_nlp()
    doc = nlp(clean_text)
    chunks = {re.sub(r"^(the|a|an)\s+", "", c.text.lower().strip()) for c in doc.noun_chunks}
    return phrase.lower() in chunks


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_keyphrases(slide: SlideContent, config: Settings) -> List[Keyphrase]:
    """Extract keyphrases from a single slide using SIFRank-style scoring.

    Adaptive filter pipeline (in order):
      1. Extract up to KEYPHRASE_MAX_CANDIDATES noun chunks via spaCy
      2. Score by cosine similarity (doc embedding ↔ phrase embedding)
      3. Drop score < KEYPHRASE_QUALITY_THRESHOLD
      4. spaCy linguistic filter  (has_noun and not all_stop and len >= 3)
      5. Noun-chunk cross-validation
      6. Assign source_type + heading boost
      7. Return sorted List[Keyphrase]

    Args:
        slide:   SlideContent from Step 2.
        config:  Settings with threshold values.

    Returns:
        List[Keyphrase] sorted by final score descending.
    """
    if not slide.clean_text.strip():
        logger.debug("Skipping empty slide %s", slide.slide_id)
        return []

    # ---- 1. Raw candidates from noun chunks ----------------------------
    candidates = _extract_candidates(slide.clean_text)
    if not candidates:
        logger.debug("No noun chunks found in %s", slide.slide_id)
        return []

    # ---- 2. SIFRank-style scoring -------------------------------------
    scores = _sifrank_scores(candidates, slide.clean_text)

    # Take top-MAX_CANDIDATES before further filtering (efficiency)
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = top_candidates[: config.keyphrase_max_candidates]

    # ---- 3. Quality threshold ----------------------------------------
    top_candidates = [(p, s) for p, s in top_candidates if s >= config.keyphrase_quality_threshold]

    # ---- 4. Linguistic filter + 5. Noun-chunk cross-validation --------
    validated: list[tuple[str, float]] = []
    for phrase, score in top_candidates:
        if not _is_valid(phrase):
            continue
        if not _in_noun_chunks(phrase, slide.clean_text):
            continue
        validated.append((phrase, score))

    # ---- 6. Source type + heading boost + appears_in ------------------
    keyphrases: list[Keyphrase] = []
    for phrase, score in validated:
        source_type = _assign_source_type(phrase, slide)
        final_score = min(score + 0.20, 1.0) if source_type == "heading" else score
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

    # Sort by final score descending
    keyphrases.sort(key=lambda k: k.score, reverse=True)
    logger.info(
        "[%s] Extracted %d keyphrases (from %d candidates)",
        slide.slide_id, len(keyphrases), len(candidates),
    )
    return keyphrases


def extract_keyphrases_all(slides: List[SlideContent], config: Settings) -> dict[str, List[Keyphrase]]:
    """Extract keyphrases for all slides. Returns {slide_id: [Keyphrase, ...]}."""
    results: dict[str, List[Keyphrase]] = {}
    for slide in slides:
        results[slide.slide_id] = extract_keyphrases(slide, config)
    return results
