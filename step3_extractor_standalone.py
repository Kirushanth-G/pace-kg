"""Step 3 — Keyphrase Extraction (standalone, no config system).

Avoids the datasets/pyarrow dependency chain by not importing core.config.Settings.
Instead, uses hardcoded constants matching CLAUDE.md / pace_kg.py exactly.

Input:  <doc_id>_preprocessed.json (Step 2 output)
Output: <doc_id>_keyphrases.json (Step 3 output)

Usage:
    python step3_extractor_standalone.py --input CS1050-L03_preprocessed.json --doc-id CS1050-L03
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants (from CLAUDE.md) ─────────────────────────────────────────────────
GLINER_MODEL           = "urchade/gliner_large-v2.1"
GLINER_THRESHOLD       = 0.35
DEDUP_SIM_THRESHOLD    = 0.85
KEYPHRASE_MAX          = 25
HEADING_BOOST          = 0.15

GLINER_LABELS = [
    "Academic Concept",
    "Theoretical Principle",
    "Technical Term",
    "Process or Method",
    "System or Framework",
    "Formula or Equation",
]

# ── Data model for preprocessed slide ───────────────────────────────────────────
@dataclass
class SlideContent:
    slide_id: str
    page_number: int
    doc_id: str
    headings: List[str]
    bullets: List[str]
    table_cells: List[str]
    captions: List[str]
    body_text: List[str]
    heading_phrases: List[str]
    clean_text: str


# ── Data model for keyphrase ───────────────────────────────────────────────────
@dataclass
class Keyphrase:
    phrase: str
    score: float
    source_type: str
    slide_id: str
    doc_id: str
    appears_in: str


# ── Load preprocessed JSON ─────────────────────────────────────────────────────
def load_preprocessed(json_path: str) -> List[SlideContent]:
    """Load preprocessed slides from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SlideContent(**item) for item in data]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _assign_source_type(phrase: str, sc: SlideContent) -> str:
    """Assign source bucket — check headings first, then bullets, table, caption, body."""
    pl = phrase.lower()
    for h in sc.headings:
        if pl in h.lower():
            return "heading"
    for b in sc.bullets:
        if pl in b.lower():
            return "bullet"
    for t in sc.table_cells:
        if pl in t.lower():
            return "table"
    for c in sc.captions:
        if pl in c.lower():
            return "caption"
    return "body"


def _find_sentence(phrase: str, clean_text: str) -> str:
    """Return the sentence from clean_text that contains the phrase."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(clean_text)
    pl = phrase.lower()
    for sent in doc.sents:
        if pl in sent.text.lower():
            return sent.text.strip()
    return clean_text[:200]


def _deduplicate(kps: List[Keyphrase]) -> List[Keyphrase]:
    """Remove near-duplicate keyphrases via SBERT cosine ≥ 0.85."""
    from sentence_transformers import SentenceTransformer, util as st_util

    if len(kps) <= 1:
        return kps

    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    sorted_kps = sorted(kps, key=lambda k: k.score, reverse=True)
    kept: List[Keyphrase] = []

    for kp in sorted_kps:
        emb = minilm.encode(kp.phrase, convert_to_tensor=True)
        is_dup = any(
            float(
                st_util.cos_sim(
                    emb, minilm.encode(k.phrase, convert_to_tensor=True)
                )
            )
            >= DEDUP_SIM_THRESHOLD
            for k in kept
        )
        if not is_dup:
            kept.append(kp)

    return kept


# ── Main keyphrase extraction ──────────────────────────────────────────────────
def extract_keyphrases_for_slide(sc: SlideContent) -> List[Keyphrase]:
    """Extract keyphrases from a single slide using GLiNER."""
    from gliner import GLiNER

    if not sc.clean_text.strip():
        return []

    # Build input text: headings first so GLiNER weights them naturally
    heading_text = " ".join(sc.headings)
    rest_text = " ".join(sc.body_text + sc.bullets + sc.table_cells + sc.captions)
    extract_text = (
        (heading_text + ". " + rest_text).strip()
        if heading_text and rest_text
        else (heading_text or rest_text)
    )

    if not extract_text.strip():
        return []

    # Load GLiNER (once per slide is fine for now; optimize later if needed)
    logger.info(f"  Loading GLiNER {GLINER_MODEL}...")
    gliner = GLiNER.from_pretrained(GLINER_MODEL)

    # GLiNER inference
    try:
        entities = gliner.predict_entities(
            extract_text,
            GLINER_LABELS,
            threshold=GLINER_THRESHOLD,
        )
    except Exception as e:
        logger.error(f"    GLiNER error on {sc.slide_id}: {e}")
        return []

    if not entities:
        return []

    # Collapse duplicate spans — keep highest score per phrase
    best: Dict[str, float] = {}
    for ent in entities:
        phrase = ent["text"].lower().strip()
        score = float(ent["score"])
        if len(phrase) < 3:
            continue
        if phrase not in best or score > best[phrase]:
            best[phrase] = score

    # Build Keyphrase objects
    kps: List[Keyphrase] = []
    for phrase, score in best.items():
        src = _assign_source_type(phrase, sc)
        final = min(score + HEADING_BOOST, 1.0) if src == "heading" else score
        app = _find_sentence(phrase, sc.clean_text)
        kps.append(
            Keyphrase(
                phrase=phrase,
                score=round(final, 4),
                source_type=src,
                slide_id=sc.slide_id,
                doc_id=sc.doc_id,
                appears_in=app,
            )
        )

    # Deduplicate near-synonyms, sort, cap
    kps = _deduplicate(kps)
    kps.sort(key=lambda k: k.score, reverse=True)
    return kps[:KEYPHRASE_MAX]


def is_pedagogical_slide(sc: SlideContent) -> bool:
    """Return False for title/ToC slides and decorative/near-empty slides."""
    word_count = len(sc.clean_text.split())
    # Skip the first slide if it has a low word count (likely a title slide)
    if sc.page_number == 1 and word_count < 40:
        return False
    # Skip decorative/empty slides throughout the PDF
    if word_count < 8:
        return False
    return True


# ── Run on all slides ──────────────────────────────────────────────────────────
def run_step3(input_json: str, output_json: str) -> None:
    """Extract keyphrases from all content slides."""
    logger.info(f"Loading preprocessed slides from {input_json}...")
    slides = load_preprocessed(input_json)
    logger.info(f"Loaded {len(slides)} slides")

    logger.info("\nExtracting keyphrases with GLiNER (this downloads ~1.5 GB on first run)...")
    keyphrases_by_slide: Dict[str, List[Keyphrase]] = {}
    skipped = 0

    for i, sc in enumerate(slides, 1):
        if not is_pedagogical_slide(sc):
            logger.info(
                f"  [{i:2d}/{len(slides)}] {sc.slide_id}: SKIPPED "
                f"(Administrative/Low-content, {len(sc.clean_text.split())} words)"
            )
            keyphrases_by_slide[sc.slide_id] = []
            skipped += 1
            continue

        logger.info(f"  [{i:2d}/{len(slides)}] {sc.slide_id}...")
        kps = extract_keyphrases_for_slide(sc)
        keyphrases_by_slide[sc.slide_id] = kps

        if kps:
            top3 = ", ".join(f"{k.phrase} ({k.score:.2f})" for k in kps[:3])
            logger.info(f"             → {len(kps)} keyphrases. Top-3: {top3}")
        else:
            logger.info(f"             → 0 keyphrases")

    total_kps = sum(len(v) for v in keyphrases_by_slide.values())
    logger.info(
        f"\nStep 3 complete:"
        f"\n  Total keyphrases: {total_kps}"
        f"\n  Slides with content: {len(slides) - skipped}/{len(slides)}"
        f"\n  Avg per content slide: {total_kps / (len(slides) - skipped):.1f if skipped < len(slides) else 0}"
    )

    # Save output
    step3_out = {sid: [asdict(k) for k in kps] for sid, kps in keyphrases_by_slide.items()}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(step3_out, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {output_json} ({Path(output_json).stat().st_size:,} bytes)")

    # Print summary table
    logger.info(f"\n{'slide_id':<12}  {'#kps':>4}  top-5 keyphrases")
    logger.info("-" * 100)
    for sc in slides:
        kps = keyphrases_by_slide.get(sc.slide_id, [])
        top5 = ", ".join(f"{k.phrase} ({k.score:.2f}/{k.source_type[0]})" for k in kps[:5])
        logger.info(f"{sc.slide_id:<12}  {len(kps):>4}  {top5}")


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="Step 3 — Keyphrase Extraction (standalone)")
    parser.add_argument(
        "--input",
        type=str,
        default="CS1050-L03_preprocessed.json",
        help="Path to preprocessed JSON (Step 2 output)",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document ID (for fallback output filename generation)",
    )
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        logger.error(f"Input file not found: {inp}")
        return 1

    # Determine output filename
    doc_id = args.doc_id or (inp.stem.replace("_preprocessed", "") if "_preprocessed" in inp.stem else inp.stem)
    out = inp.parent / f"{doc_id}_keyphrases.json"

    try:
        run_step3(str(inp), str(out))
    except Exception as e:
        logger.exception(f"Step 3 failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
