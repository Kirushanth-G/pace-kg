"""Quick test: run Step 1 (Marker parser) + Step 2 (preprocessor) on a real PDF."""
import sys
import os

# Make backend the root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from pipeline.step1_marker.parser import parse_pdf
from pipeline.step2_preprocessor.cleaner import preprocess_slides

PDF = "/home/kirus/Documents/Projects/Edu-KG/main.pdf"
DOC_ID = "test-doc-001"

print(f"\n{'='*60}")
print("STEP 1 — Marker PDF Parsing")
print('='*60)
slides_md = parse_pdf(PDF, DOC_ID)
print(f"✓ Parsed {len(slides_md)} slides\n")

for s in slides_md[:3]:
    print(f"  [{s.slide_id}] page={s.page_number}  chars={len(s.raw_markdown)}")
    print(f"  Preview: {s.raw_markdown[:120].replace(chr(10), ' ')!r}")
    print()

print(f"\n{'='*60}")
print("STEP 2 — Markdown Preprocessor")
print('='*60)
contents = preprocess_slides(slides_md)
print(f"✓ Preprocessed {len(contents)} slides\n")

for c in contents[:3]:
    print(f"  [{c.slide_id}]  headings={len(c.headings)}  bullets={len(c.bullets)}  "
          f"body={len(c.body_text)}  clean_text_len={len(c.clean_text)}")
    if c.heading_phrases:
        print(f"    Heading phrases: {c.heading_phrases[:3]}")
    print(f"    clean_text preview: {c.clean_text[:120]!r}")
    print()

print("✓ Steps 1 & 2 complete.")
