# COLAB_MARKER_GUIDE.md — Parse Image-Heavy PDFs with Marker on Google Colab

> Use this guide ONLY when your lecture PDF contains important text embedded inside
> images/diagrams that pymupdf4llm cannot extract. For text-based PDFs (headings,
> bullets, tables), skip this — pymupdf4llm works fine on your local 8GB machine.

---

## When to use this

| Your PDF has... | Use |
|---|---|
| Mostly text, headings, bullets, tables | `parse_pdf()` locally (pymupdf4llm) |
| Diagrams/flowcharts with text labels inside images | This guide (Marker on Colab) |
| Mathematical notation as images | This guide (Marker on Colab) |

---

## What this produces

Running this guide on Google Colab outputs a single JSON file:
`{your_pdf_name}_parsed.json`

That JSON is a list of `SlideMarkdown` dicts — identical format to what
`parse_pdf()` produces locally. You download it, drop it into `_cache/`, and
call `load_parsed_json()` instead of `parse_pdf()`. One-time cost per PDF.

---

## Step-by-step instructions

### 1. Open a Google Colab notebook

Go to [colab.google](https://colab.google) → New notebook.

Free tier gives ~12–16 GB RAM (T4 GPU or CPU) — enough to run all 5 Marker models.

---

### 2. Install Marker

Paste and run in a Colab cell:

```python
!pip install marker-pdf -q
```

Wait ~2–3 minutes for install + model downloads (surya models auto-download on first run).

---

### 3. Upload your PDF

```python
from google.colab import files
uploaded = files.upload()   # click "Choose Files", select your PDF
pdf_path = list(uploaded.keys())[0]
print(f"Uploaded: {pdf_path}")
```

---

### 4. Run the parser and export JSON

Paste this entire cell and run it:

```python
import hashlib, json
from pathlib import Path
from dataclasses import asdict, dataclass

@dataclass
class SlideMarkdown:
    slide_id: str
    page_number: int
    raw_markdown: str
    doc_id: str

# ── Config ──────────────────────────────────────────────────────────────────
PDF_PATH = pdf_path                           # from the upload cell above
DOC_ID   = Path(pdf_path).stem               # use filename as doc_id (or change this)
PAGE_SEP = "-" * 48                          # Marker 1.x default page separator

# ── Run Marker ───────────────────────────────────────────────────────────────
print("Loading Marker models (first run downloads ~4 GB)…")
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

model_dict = create_model_dict()
converter  = PdfConverter(
    artifact_dict=model_dict,
    config={"paginate_output": True, "page_separator": PAGE_SEP},
)
rendered      = converter(PDF_PATH)
full_markdown = rendered.markdown

# ── Split into pages ─────────────────────────────────────────────────────────
raw_pages = [p.strip() for p in full_markdown.split(PAGE_SEP)]

slides = []
for idx, page_md in enumerate(raw_pages, start=1):
    page_md = page_md.strip()
    if not page_md:
        continue
    slides.append(SlideMarkdown(
        slide_id    = f"slide_{idx:03d}",
        page_number = idx,
        raw_markdown= page_md,
        doc_id      = DOC_ID,
    ))

print(f"Parsed {len(slides)} slides from {PDF_PATH}")

# ── Save JSON ────────────────────────────────────────────────────────────────
out_path = f"{Path(PDF_PATH).stem}_parsed.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump([asdict(s) for s in slides], f, ensure_ascii=False, indent=2)

print(f"Saved: {out_path}")
for s in slides[:3]:
    print(f"\n{'='*60}")
    print(f"[{s.slide_id}] page {s.page_number}")
    print(s.raw_markdown[:300])
```

---

### 5. Download the JSON

```python
from google.colab import files
files.download(out_path)
```

A `{name}_parsed.json` file will be saved to your Downloads folder.

---

### 6. Use it locally in PACE-KG

1. Copy the downloaded JSON file into:
   ```
   pace-kg/backend/pipeline/step1_marker/_cache/
   ```

2. In your code (or test script), call `load_parsed_json()` instead of `parse_pdf()`:
   ```python
   from pipeline.step1_marker.parser import load_parsed_json, preprocess_slides

   # Replace parse_pdf() with this:
   slides = load_parsed_json(
       "_cache/my_lecture_parsed.json",
       doc_id="my-doc-001"
   )
   ```

3. Continue with Step 2 onwards — everything else is identical.

---

## How long does it take?

| PDF size | Colab free (CPU) | Colab free (T4 GPU) |
|---|---|---|
| 10-slide lecture | ~2 min | ~45 sec |
| 30-slide lecture | ~6 min | ~2 min |
| 80-slide deck | ~15 min | ~5 min |

Model downloads happen only once per Colab session (~4 GB total, ~5 min first time).

---

## Tips

- **Re-use the session**: If you have multiple PDFs, keep the Colab tab open.
  Models are already loaded — each additional PDF takes only parsing time.

- **Colab Pro is not needed**: Free tier has enough RAM (12–16 GB) for Marker.
  If the session disconnects, just re-run cells 1–3 (models re-download in the background).

- **Naming the DOC_ID**: Change `DOC_ID = Path(pdf_path).stem` to any string you want.
  It can be overridden when calling `load_parsed_json(path, doc_id="...")`.

- **The JSON is the cache**: Once downloaded, `load_parsed_json()` never re-parses
  the PDF. The SHA-256 caching in `parse_pdf()` also accepts it if you copy it to
  `_cache/` with the correct hash filename — but using `load_parsed_json()` directly
  is simpler.
