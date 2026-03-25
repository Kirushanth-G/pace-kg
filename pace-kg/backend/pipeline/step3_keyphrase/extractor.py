"""Step 3 — Keyphrase Extraction (SIFRank-style with sentence-transformers).

Implements the SIFRank algorithm spirit using:
  - spaCy en_core_web_sm  : candidate extraction (noun chunks) + linguistic validation
  - all-MiniLM-L6-v2      : phrase and document embeddings (80MB — 8GB friendly)

Algorithm (mirrors SIFRankSqueezeBERT from coursemapper-kg):
  1. Extract candidate noun chunks from clean_text (excluding code_lines)
  2. Score by cosine similarity to document embedding  (SIFRank-style)
  3. Apply CLAUDE.md adaptive filter pipeline (quality, linguistic, noun-chunk, source, heading boost)
  4. Deduplicate near-duplicate phrases via SBERT similarity
  5. Return List[Keyphrase] sorted by final score descending

Fixes applied vs original:
  FIX 1 — table cells now included in SIFRank input text
  FIX 2 — language-agnostic code line detection (no Java-specific keywords)
  FIX 3 — code lines routed to code_lines bucket, excluded from SIFRank input
  FIX 4 — near-duplicate phrase deduplication with SBERT
  FIX 5 — constructor signature filter (language-agnostic)
  FIX 6 — phrase cleaner strips slide-number artifacts and code fence markers
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
from pipeline.utils import is_code_line as _is_code_line

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
# FIX 2 — Language-agnostic code line detection (now in pipeline/utils.py)
# ---------------------------------------------------------------------------
# _is_code_line is imported from pipeline.utils above — keeping the name
# _is_code_line (with underscore) for backward compatibility within this module.


# ---------------------------------------------------------------------------
# FIX 6 — Phrase cleaner
# ---------------------------------------------------------------------------
def _clean_phrase(phrase: str) -> str:
    """Strip slide-number artifacts, code fences, and stray punctuation.

    Handles patterns like:
      "hashmap implementation 1/2 jvm"  →  "hashmap implementation jvm"
      "summary hash-based collections"  →  "hash-based collections"
      "```import java.util"             →  removed entirely (returns "")
    """
    # Remove code fence markers
    if "```" in phrase:
        return ""

    # Remove slide number tags like {5}, {12}
    phrase = re.sub(r"\{\d+\}", "", phrase)

    # Remove version/part markers like 1/2, 2/2
    phrase = re.sub(r"\b\d+/\d+\b", "", phrase)

    # Remove leading section words that add no concept value
    phrase = re.sub(r"^(summary|overview|introduction|conclusion)\s+",
                    "", phrase, flags=re.IGNORECASE)

    # Strip stray punctuation at edges
    phrase = phrase.strip(" .,;:-/\\")

    return phrase.strip()


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------
def _extract_candidates(sifrank_text: str) -> list[str]:
    """Extract unique lowercase noun-chunk candidates from text via spaCy.

    Uses sifrank_text which excludes code_lines (FIX 3).
    """
    nlp = _get_nlp()
    doc = nlp(sifrank_text)
    seen: set[str] = set()
    candidates: list[str] = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # Remove leading determiners
        phrase = re.sub(r"^(the|a|an)\s+", "", phrase).strip()

        # FIX 6: clean artifact patterns before accepting
        phrase = _clean_phrase(phrase)
        if not phrase:
            continue

        # Skip phrases that start with a digit — numbered-list artifacts
        # e.g. "2. map interface", "1. int hash value", "4. constructor hashmap"
        if re.match(r"^\d", phrase):
            continue

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
    embeddings = model.encode(
        all_texts, convert_to_tensor=True, show_progress_bar=False
    )

    doc_emb   = embeddings[0]
    cand_embs = embeddings[1:]

    scores: dict[str, float] = {}
    for phrase, cand_emb in zip(candidates, cand_embs):
        scores[phrase] = float(
            util.cos_sim(doc_emb.unsqueeze(0), cand_emb.unsqueeze(0))
        )
    return scores


# ---------------------------------------------------------------------------
# Adaptive filter helpers
# ---------------------------------------------------------------------------
def _is_valid(phrase: str) -> bool:
    """spaCy linguistic filter: must contain a noun, not be all-stop, len >= 3."""
    nlp = _get_nlp()
    doc = nlp(phrase)
    has_noun  = any(t.pos_ in ("NOUN", "PROPN") for t in doc)
    all_stop  = all(t.is_stop for t in doc)
    return has_noun and not all_stop and len(phrase.strip()) >= 3


def _in_noun_chunks(phrase: str, sifrank_text: str) -> bool:
    """Noun-chunk cross-validation: phrase must appear as a spaCy noun chunk."""
    nlp   = _get_nlp()
    doc   = nlp(sifrank_text)
    chunks = {
        re.sub(r"^(the|a|an)\s+", "", c.text.lower().strip())
        for c in doc.noun_chunks
    }
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
    pl  = phrase.lower()
    for sent in doc.sents:
        if pl in sent.text.lower():
            return sent.text.strip()
    return clean_text[:200]


# ---------------------------------------------------------------------------
# FIX 4 — Near-duplicate deduplication
# ---------------------------------------------------------------------------
def _deduplicate(keyphrases: list[Keyphrase], sim_threshold: float = 0.85) -> list[Keyphrase]:
    """Remove near-duplicate keyphrases keeping the highest-scored variant.

    Example: keeps "composite data structures" (score 0.68) and drops
    "collections composite data structures" (0.58), "most common composite
    data structures" (0.59), "composite data structure" (0.56) — all the
    same concept expressed differently.

    Uses SBERT all-MiniLM-L6-v2 (already loaded — no extra cost).
    Runs pairwise comparison on the small per-slide keyphrase list (typically
    10-25 items) so O(n²) is not a concern here.
    """
    if len(keyphrases) <= 1:
        return keyphrases

    model = get_minilm()

    # Sort descending by score so we always keep the best-scoring variant
    sorted_kps = sorted(keyphrases, key=lambda k: k.score, reverse=True)
    kept: list[Keyphrase] = []

    for kp in sorted_kps:
        emb = model.encode(kp.phrase, convert_to_tensor=True)
        is_dup = False
        for kept_kp in kept:
            kept_emb = model.encode(kept_kp.phrase, convert_to_tensor=True)
            sim = float(util.cos_sim(emb, kept_emb))
            if sim >= sim_threshold:
                is_dup = True
                logger.debug(
                    "Dedup: '%s' (%.3f) is duplicate of '%s' (sim=%.3f)",
                    kp.phrase, kp.score, kept_kp.phrase, sim,
                )
                break
        if not is_dup:
            kept.append(kp)

    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_keyphrases(slide: SlideContent, config: Settings) -> List[Keyphrase]:
    """Extract keyphrases from a single slide using SIFRank-style scoring.

    Adaptive filter pipeline (in order):
      1.  Build SIFRank input text (headings + body + bullets + tables + captions)
          excluding code_lines  [FIX 1, FIX 3]
      2.  Extract up to KEYPHRASE_MAX_CANDIDATES noun chunks via spaCy
      3.  Clean phrase artifacts (slide numbers, code fences)  [FIX 6]
      4.  Score by cosine similarity (doc embedding ↔ phrase embedding)
      5.  Drop score < KEYPHRASE_QUALITY_THRESHOLD
      6.  spaCy linguistic filter (has_noun, not all_stop, len >= 3)
      7.  Noun-chunk cross-validation
      8.  Assign source_type + heading boost (+0.20)
      9.  Deduplicate near-duplicates via SBERT  [FIX 4]
      10. Return sorted List[Keyphrase]

    Args:
        slide:   SlideContent from Step 2 (must have code_lines field).
        config:  Settings with threshold values.

    Returns:
        List[Keyphrase] sorted by final score descending.
    """
    if not slide.clean_text.strip():
        logger.debug("Skipping empty slide %s", slide.slide_id)
        return []

    # ---- FIX 1 + FIX 3: build SIFRank input without code lines --------
    # table_cells included here (they were excluded in the original)
    # code_lines excluded so constructor signatures / code tokens don't
    # become keyphrase candidates.
    # body_text is also filtered through _is_code_line() as a safety net
    # for any lines that weren't caught during Step 2 routing.
    clean_body    = [line for line in slide.body_text if not _is_code_line(line)]
    clean_bullets  = [b    for b    in slide.bullets   if not _is_code_line(b)]

    # Use ". " to separate headings from the rest so that spaCy treats them as
    # different sentences — prevents cross-boundary noun chunks like
    # "LinkedList Class LinkedList class" when the first bullet starts with
    # the same words as the heading.
    heading_text = " ".join(slide.headings)
    rest_parts   = clean_body + clean_bullets + slide.table_cells + slide.captions
    rest_text    = " ".join(rest_parts)

    if heading_text and rest_text:
        sifrank_text = heading_text + ". " + rest_text
    else:
        sifrank_text = (heading_text or rest_text)

    if not sifrank_text:
        logger.debug("No non-code text in %s", slide.slide_id)
        return []

    # ---- 2. Raw candidates from noun chunks ----------------------------
    candidates = _extract_candidates(sifrank_text)
    if not candidates:
        logger.debug("No noun chunks found in %s", slide.slide_id)
        return []

    # ---- 3. SIFRank-style scoring -------------------------------------
    scores = _sifrank_scores(candidates, sifrank_text)

    # Take top MAX_CANDIDATES before further filtering (efficiency)
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = top_candidates[: config.keyphrase_max_candidates]

    # ---- 4. Quality threshold ----------------------------------------
    top_candidates = [
        (p, s) for p, s in top_candidates
        if s >= config.keyphrase_quality_threshold
    ]

    # ---- 5. Linguistic filter + 6. Noun-chunk cross-validation --------
    validated: list[tuple[str, float]] = []
    for phrase, score in top_candidates:
        if not _is_valid(phrase):
            continue
        if not _in_noun_chunks(phrase, sifrank_text):
            continue
        validated.append((phrase, score))

    # ---- 7. Source type + heading boost + appears_in ------------------
    # Note: _find_sentence uses full clean_text (includes code context)
    # so LLM in Step 4 gets the complete sentence even if it's in code
    keyphrases: list[Keyphrase] = []
    for phrase, score in validated:
        source_type = _assign_source_type(phrase, slide)
        final_score = min(score + 0.20, 1.0) if source_type == "heading" else score
        appears_in  = _find_sentence(phrase, slide.clean_text)

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

    # ---- FIX 4: deduplicate near-duplicates ---------------------------
    keyphrases = _deduplicate(keyphrases, sim_threshold=0.85)

    # Final sort by score descending
    keyphrases.sort(key=lambda k: k.score, reverse=True)

    logger.info(
        "[%s] Extracted %d keyphrases (from %d candidates)",
        slide.slide_id, len(keyphrases), len(candidates),
    )
    return keyphrases


def extract_keyphrases_all(
    slides: List[SlideContent], config: Settings
) -> dict[str, List[Keyphrase]]:
    """Extract keyphrases for all slides. Returns {slide_id: [Keyphrase, ...]}."""
    results: dict[str, List[Keyphrase]] = {}
    for slide in slides:
        results[slide.slide_id] = extract_keyphrases(slide, config)
    return results