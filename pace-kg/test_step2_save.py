"""Test Step 1 + Step 2, print full results in the terminal, and save to JSON.

Output file: pace-kg/test_output/step2_results.json

Run:
    cd pace-kg
    source backend/.venv/bin/activate
    python test_step2_save.py
"""
import sys, os, json
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from pipeline.step1_marker.parser import parse_pdf
from pipeline.step2_preprocessor.cleaner import preprocess_slides

# ── Config ────────────────────────────────────────────────────────────────────
PDF      = "/home/kirus/Documents/Projects/Edu-KG/main.pdf"
DOC_ID   = "test-doc-001"
OUT_DIR  = Path(__file__).parent / "test_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT      = OUT_DIR / "step2_results.json"

# ── Step 1 ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 1 — pymupdf4llm PDF Parsing")
print("="*60)
slides_md = parse_pdf(PDF, DOC_ID)
print(f"✓ Parsed {len(slides_md)} slides\n")

# Save Step 1 raw markdown — one .md file per slide
md_dir = OUT_DIR / "step1_raw_markdown"
md_dir.mkdir(parents=True, exist_ok=True)
for s in slides_md:
    (md_dir / f"{s.slide_id}.md").write_text(s.raw_markdown, encoding="utf-8")
print(f"  Step 1 raw markdown saved → {md_dir}/ ({len(slides_md)} .md files)\n")

# ── Step 2 ────────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print("STEP 2 — Markdown Preprocessor")
print("="*60)
contents = preprocess_slides(slides_md)
print(f"✓ Preprocessed {len(contents)} slides\n")

# ── Full terminal output ───────────────────────────────────────────────────────
for c in contents:
    print(f"┌─ [{c.slide_id}]  page={c.page_number}")
    print(f"│  headings    ({len(c.headings)}): {c.headings}")
    print(f"│  bullets     ({len(c.bullets)}): {c.bullets}")
    print(f"│  table_cells ({len(c.table_cells)}): {c.table_cells}")
    print(f"│  captions    ({len(c.captions)}): {c.captions}")
    print(f"│  body_text   ({len(c.body_text)}): {c.body_text}")
    print(f"│  heading_phrases: {c.heading_phrases}")
    print(f"└  clean_text  ({len(c.clean_text)} chars): {c.clean_text!r}")
    print()

# ── Save to JSON ───────────────────────────────────────────────────────────────
results = []
for c in contents:
    results.append({
        "slide_id":        c.slide_id,
        "page_number":     c.page_number,
        "doc_id":          c.doc_id,
        "headings":        c.headings,
        "bullets":         c.bullets,
        "table_cells":     c.table_cells,
        "captions":        c.captions,
        "body_text":       c.body_text,
        "heading_phrases": c.heading_phrases,
        "clean_text":      c.clean_text,
        "stats": {
            "n_headings":    len(c.headings),
            "n_bullets":     len(c.bullets),
            "n_table_cells": len(c.table_cells),
            "n_captions":    len(c.captions),
            "n_body":        len(c.body_text),
            "clean_text_len": len(c.clean_text),
        }
    })

with OUT.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"{'='*60}")
print(f"✓ Saved results for {len(results)} slides → {OUT}")
print(f"  Open test_output/step2_results_cs.json to inspect full output.")
