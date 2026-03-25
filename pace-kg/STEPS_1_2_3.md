# PACE-KG — Steps 1–4 Implementation Notes

---

## Step 1 — PDF Parsing (`pipeline/step1_marker/parser.py`)

- **Tool:** `pymupdf4llm.to_markdown(file_path, page_chunks=True)`
- Returns list of dicts with `"text"` (markdown) and `"metadata"["page"]` (0-indexed)
- **Caching:** SHA-256 hash of PDF bytes → JSON cache; skips re-parse on hit
- **slide_id format:** zero-padded `slide_001`, `slide_002`, …
- Output dataclass: `SlideMarkdown(slide_id, page_number, raw_markdown, doc_id)`
- Does **not** extract text from images — use Marker/Colab workflow for image-heavy PDFs

---

## Step 2 — Markdown Preprocessor (`pipeline/step2_preprocessor/cleaner.py`)

### For pymupdf4llm output (`preprocess_slide`)

**Stage 1 — Structural bucket parsing (mistune):**
- `#` / `##` lines → `headings`
- `-` / `*` lines → `bullets`
- `| cell |` rows → `table_cells`
- `>` lines → `captions`
- Everything else → `body_text`

**Stage 2 — Noise pattern removal:**
```python
NOISE_PATTERNS = [
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*\d+\s*$",
    r"^©.*",
    r"^\s*(references|bibliography)\s*$",
    r"https?://\S+",
    r"^\s*\[\d+\]",
]
```

**Stage 3 — Cross-slide repetition filter:**
- Any text block appearing in > 50% of slides → discarded from all slides

**Stage 4 — Assemble `clean_text`:**
```python
clean_text = " ".join(headings + body_text + bullets + table_cells + captions)
```

`heading_phrases` stored separately for Step 3 heading boost.

---

### For Marker JSON input (`load_preprocessed_json`)

Marker pre-processes image-heavy PDFs via Colab. JSON is loaded directly.

**Heading deduplication fix:**
```python
heading_set = {h.lower().strip() for h in item.get("headings", [])}
for line in body_text:
    if line.lower().strip() in heading_set:
        continue  # skip exact heading duplicates
    for h in heading_set:
        if line.lower().startswith(h + " ") or line.lower().startswith(h + "\t"):
            line = line[len(h):].strip()  # strip heading prefix from body line
```
*Reason: Marker sometimes embeds heading text at start of first body paragraph.*

**Code detection routing (`pipeline/utils.py → is_code_line()`):**

Lines matching any of 8 signals → routed to `code_lines`, excluded from NLP:
1. Code fence (` ``` `)
2. High punctuation density: `{}();=<>[]` count / len > 0.15
3. camelCase token (`[a-z][A-Z]`)
4. snake_case token (`[a-z]_[a-z]`)
5. Assignment / comparison operators (`==`, `!=`, `+=`, etc.)
6. Line ends with `;`, `{`, or `}`
7. Comment patterns: `//`, `#`, `--`, `/*`
8. Function/constructor call pattern: `word(` or `)` at end

---

## Step 3 — Keyphrase Extraction (`pipeline/step3_keyphrase/extractor.py`)

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (SIFRank-style scoring)  
**Noun chunk extraction:** spaCy `en_core_web_sm`

### sifrank_text assembly (critical fix)

```python
clean_bullets = [b for b in slide.bullets if not _is_code_line(b)]
clean_body    = [b for b in slide.clean_body if not _is_code_line(b)]
heading_text  = " ".join(slide.headings)
rest_text     = " ".join(clean_body + clean_bullets + slide.table_cells + slide.captions)
# ". " separator prevents cross-boundary noun chunks
sifrank_text  = heading_text + ". " + rest_text  # if both non-empty
```
*Without `". "`, spaCy merges heading + first bullet into doubled chunks like `"linkedlist class linkedlist class"`.*

### Adaptive filter pipeline (in order)

1. Extract up to `KEYPHRASE_MAX_CANDIDATES = 30` noun chunks via spaCy
2. **Digit-prefix filter:** `if re.match(r"^\d", phrase): continue`  
   *(removes numbered-list artifacts: `"1. int hash value"`, `"2. map interface"`)*
3. Score < `KEYPHRASE_QUALITY_THRESHOLD = 0.3` → dropped
4. **spaCy linguistic filter:**
   ```python
   has_noun = any(t.pos_ in ["NOUN", "PROPN"] for t in doc)
   all_stop  = all(t.is_stop for t in doc)
   valid = has_noun and not all_stop and len(phrase.strip()) >= 3
   ```
5. **Noun-chunk cross-validation:** phrase must appear in spaCy noun chunks of `clean_text`
6. **Source-type assignment:** check buckets in order: heading → bullet → table → caption → body
7. **Heading boost:** `if source_type == "heading": score = min(score + 0.20, 1.0)`

### Output
`Keyphrase(phrase, score, source_type, slide_id, doc_id, appears_in)`

---

## Key Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Constructor sigs extracted (`constructor hashmap(map` 0.706) | Raw bullets fed into sifrank_text | `clean_bullets` code filter |
| Heading doubling (`linkedlist class linkedlist class` 0.768) | Heading + first bullet merged → cross-boundary noun chunk | `". "` sentence separator in sifrank_text |
| Numbered-list artifacts (`2. map interface`) | Digit-prefixed noun chunks | `re.match(r"^\d", phrase)` filter |
| Code body_text leaks (`+hashcode+" index`) | Marker wraps code in triple-backtick body_text | `is_code_line()` routing in cleaner + extractor |
| Heading repeated in body_text | Marker embeds heading at start of first body paragraph | Exact-match + prefix-strip dedup in `load_preprocessed_json()` |

---

## Step 4 — LLM Triple Extraction (`pipeline/step4_llm_extraction/`)

### Files
- `prompts.py` — `SYSTEM_PROMPT` and `USER_PROMPT` (verbatim from CLAUDE.md)
- `extractor.py` — `TripleExtractor`: calls Groq, auto-fallbacks on HTTP 429
- `validator.py` — `TripleValidator`: all 3 mandatory validation layers

### Extractor
- Sends keyphrase list + `clean_text` to `llama-3.3-70b-versatile` via `langchain_groq`
- `temperature=0`, `response_format: json_object`
- On HTTP 429: retries up to 3×, switching to `llama-3.1-8b-instant` fallback
- Returns raw `list[dict]` (unvalidated)

### Validator — 3 layers (all must pass)

| Layer | Check |
|-------|-------|
| 1 — Anchor | subject and object in keyphrase set (lowercase); subject ≠ object; relation in allowed 8 |
| 2 — Evidence | SBERT cosine sim(evidence, slide_text) ≥ `EVIDENCE_SIMILARITY_THRESHOLD` (0.75) |
| 3 — Confidence | LLM confidence ≥ `TRIPLE_CONFIDENCE_THRESHOLD` (0.70) |

SBERT loaded via `get_sbert()` singleton — never instantiated per call.

### Test results (CS1050-L03, 3 slides)
- slide_003: 3/5 passed (60%)
- slide_004: 0/3 passed (keyphrase anchor mismatch)
- slide_005: 6/8 passed (75%)
- Output: `test_output/step4_triples.json`
