"""Test Step 3 — keyphrase extraction using Marker-preprocessed JSON.

Input:  CS1050-L03_preprocessed.json  (Step 2 already done by Marker pipeline)
Output: test_output/step3_results.json

Runs full extraction on first 3 content slides (slides 003-005) for quick
manual inspection per CLAUDE.md Week 2 guidance.
"""
import json
import logging
import sys
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from core.config import Settings
from pipeline.step2_preprocessor.cleaner import load_preprocessed_json
from pipeline.step3_keyphrase.extractor import extract_keyphrases

# ── config ───────────────────────────────────────────────────────────────────
MARKER_JSON = Path("/home/kirus/Documents/Projects/Edu-KG/CS1050-L03_preprocessed.json")
DOC_ID      = "CS1050-L03"
OUT_DIR     = ROOT / "test_output"
OUT_JSON    = OUT_DIR / "step3_keyphrases_all.json"
N_SLIDES    = None        # None = run on ALL content slides

config = Settings()

# ── Step 2: load pre-processed slides ────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Loading Marker-preprocessed JSON")
print("=" * 60)

slides = load_preprocessed_json(str(MARKER_JSON), DOC_ID)
# Skip title/diagram-only slides — require meaningful body content (bullets or body_text)
content_slides = [
    s for s in slides
    if len(s.bullets) >= 2 or len(s.clean_text.strip()) > 200
]
target_slides  = content_slides[:N_SLIDES]

print(f"✓ Loaded {len(slides)} slides total, {len(content_slides)} with content")
print(f"  Running Step 3 on: {[s.slide_id for s in target_slides]}")

# ── Step 3: keyphrase extraction ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Keyphrase Extraction (SIFRank + MiniLM-L6-v2)")
print("=" * 60)
print("  Loading model (all-MiniLM-L6-v2) — first run downloads ~80MB …")

results = []

for slide in target_slides:
    keyphrases = extract_keyphrases(slide, config)

    print(f"\n┌─ [{slide.slide_id}]  page={slide.page_number}")
    print(f"│  headings  : {slide.headings}")
    print(f"│  keyphrases ({len(keyphrases)}):")
    for kp in keyphrases:
        boost_tag = " [+heading]" if kp.source_type == "heading" else ""
        print(f"│    [{kp.score:.3f}] ({kp.source_type}{boost_tag}) {kp.phrase!r}")
    print(f"└  clean_text ({len(slide.clean_text)} chars): {slide.clean_text[:120]!r}…")

    results.append({
        "slide_id":    slide.slide_id,
        "page_number": slide.page_number,
        "doc_id":      slide.doc_id,
        "headings":    slide.headings,
        "clean_text":  slide.clean_text,
        "keyphrases": [
            {
                "phrase":      kp.phrase,
                "score":       kp.score,
                "source_type": kp.source_type,
                "appears_in":  kp.appears_in,
            }
            for kp in keyphrases
        ],
    })

# ── save results ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'=' * 60}")
print(f"✓ Saved results for {len(results)} slides → {OUT_JSON}")
print("  Open test_output/step3_results.json to inspect full output.")
