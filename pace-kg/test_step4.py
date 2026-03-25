"""Test Step 4 — LLM Triple Extraction (extractor + validator).

Input:  CS1050-L03_preprocessed.json  (Step 2 done externally by Marker)
        test_output/step3_keyphrases_all.json  (Step 3 already run)
Output: test_output/step4_triples.json

Runs extraction + 3-layer validation on the first 3 content slides and prints
a detailed report so the output can be inspected manually (per CLAUDE.md Week 2).

Environment: requires GROQ_API_KEY in .env or environment variable.
"""

import json
import logging
import os
import sys
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))

# Load .env so GROQ_API_KEY is available
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "backend" / ".env")
except ImportError:
    pass  # dotenv not installed — rely on environment variable

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

from core.config import Settings
from api.models.concept import Keyphrase
from pipeline.step2_preprocessor.cleaner import load_preprocessed_json, SlideContent
from pipeline.step3_keyphrase.extractor import extract_keyphrases
from pipeline.step4_llm_extraction.extractor import TripleExtractor
from pipeline.step4_llm_extraction.validator import TripleValidator

# ── paths ────────────────────────────────────────────────────────────────────
MARKER_JSON = Path("/home/kirus/Documents/Projects/Edu-KG/CS1050-L03_preprocessed.json")
DOC_ID = "CS1050-L03"
OUT_DIR = ROOT / "test_output"
OUT_JSON = OUT_DIR / "step4_triples.json"
N_SLIDES = None  # None = all content slides

config = Settings()

# ── GROQ_API_KEY guard ────────────────────────────────────────────────────────
if not config.groq_api_key:
    # Try loading from apikey.txt in project root as fallback
    apikey_file = Path("/home/kirus/Documents/Projects/Edu-KG/apikey.txt")
    if apikey_file.exists():
        first_line = apikey_file.read_text().strip().splitlines()[0].strip()
        if first_line.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = first_line
            config = Settings()  # reload with env var

if not config.groq_api_key:
    print("ERROR: GROQ_API_KEY not set. Put it in pace-kg/backend/.env or export it.")
    sys.exit(1)

print(f"GROQ_API_KEY: gsk_...{config.groq_api_key[-6:]}")

# ── Step 2: load pre-processed slides ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Loading Marker-preprocessed JSON")
print("=" * 60)
slides: list[SlideContent] = load_preprocessed_json(str(MARKER_JSON), DOC_ID)
content_slides = [
    s for s in slides if len(s.bullets) >= 2 or len(s.clean_text.strip()) > 200
]
target_slides = content_slides[:N_SLIDES]
print(f"Loaded {len(slides)} slides, {len(content_slides)} with content")
print(f"Running on: {[s.slide_id for s in target_slides]}")

# ── Step 3: keyphrase extraction ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Keyphrase Extraction (reusing Step 3 logic)")
print("=" * 60)
keyphrases_by_slide: dict[str, list[Keyphrase]] = {}
for slide in target_slides:
    kps = extract_keyphrases(slide, config)
    keyphrases_by_slide[slide.slide_id] = kps
    print(f"  [{slide.slide_id}] {len(kps)} keyphrases: {[k.phrase for k in kps[:5]]}")

# ── Step 4: LLM triple extraction + validation ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — LLM Triple Extraction + 3-Layer Validation")
print("=" * 60)

extractor = TripleExtractor(config)
validator = TripleValidator(config)

results = []

for slide in target_slides:
    keyphrases = keyphrases_by_slide[slide.slide_id]

    print(f"\n┌─ [{slide.slide_id}]  keyphrases={len(keyphrases)}")
    print(f"│  Calling LLM …")

    # Step 4a: LLM extraction
    raw_dicts = extractor.extract(keyphrases, slide)
    print(f"│  LLM returned {len(raw_dicts)} raw triple(s)")

    # Step 4b: 3-layer validation
    triples = validator.validate_all(
        raw_dicts,
        keyphrases,
        slide.clean_text,
        slide.slide_id,
        DOC_ID,
    )

    print(f"│  Validated {len(triples)}/{len(raw_dicts)} triple(s)")
    for t in triples:
        print(
            f"│    [{t.confidence:.2f}] ({t.subject}) --[{t.relation}]--> ({t.object})"
        )
        print(f"│           evidence: {t.evidence[:80]!r}…")

    if not triples:
        print("│    (no triples passed validation)")

    print("└" + "─" * 58)

    results.append(
        {
            "slide_id": slide.slide_id,
            "page_number": slide.page_number,
            "doc_id": DOC_ID,
            "keyphrases": [k.phrase for k in keyphrases],
            "raw_count": len(raw_dicts),
            "raw_triples": raw_dicts,
            "validated_count": len(triples),
            "triples": [
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                    "evidence": t.evidence,
                    "confidence": t.confidence,
                    "source": t.source,
                }
                for t in triples
            ],
        }
    )

# ── save results ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

total_raw = sum(r["raw_count"] for r in results)
total_pass = sum(r["validated_count"] for r in results)

print(f"\n{'=' * 60}")
print(
    f"Summary: {total_pass}/{total_raw} triples passed validation "
    f"across {len(results)} slides"
)
print(f"Saved → {OUT_JSON}")
print("Open test_output/step4_triples.json to inspect the full output.")
