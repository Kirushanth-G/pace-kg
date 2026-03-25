# PACE-KG Project Handover
**Date:** March 2026  
**Project:** PACE-KG — Pedagogically-Aware, Citation-Evidenced Knowledge Graph  
**Status:** Steps 1–4 implemented and tested. Steps 1 & 2 actively being debugged. Step 3 (GLiNER) and Step 4 (LLM triples) complete but pending re-run on clean data.

---

## 1. What This Project Is

A research pipeline that converts lecture slide PDFs into structured knowledge graphs (subject → relation → object triples). Being developed as an academic paper comparing against the baseline paper:

> **Ain et al. 2025** — "An Optimized Pipeline for Automatic Educational Knowledge Graph Construction"

The key novel contributions over the baseline are:
- **Marker** (deep-learning PDF parser) instead of basic text extraction
- **GLiNER** (zero-shot NER) instead of SIFRank for keyphrase extraction
- **3-layer triple validation** (anchor + evidence + confidence) instead of unconstrained LLM output
- **Pedagogical-awareness**: code lines excluded from keyphrase pool, heading boost, cross-slide repetition filter

---

## 2. Pipeline Architecture (4 Steps)

```
PDF → [Step 1: Marker Parse] → SlideMarkdown JSON
    → [Step 2: Preprocessor] → SlideContent JSON  (buckets: headings/bullets/body/table/code)
    → [Step 3: GLiNER]       → Keyphrase JSON      (phrase, score, source_type, appears_in)
    → [Step 4: LLM Triples]  → Triple JSON         (subject, relation, object, evidence, confidence)
```

**All code lives in a single Google Colab notebook: `Edu_kg_gliner.ipynb`**  
Colab link (may need re-saving): `https://colab.research.google.com/drive/13U8c56Ws08HGZMrwz-DTJAAp4jB5TIKe`

**Runtime required:** T4 GPU (Marker + GLiNER both need GPU)  
**Colab tier needed:** Pro (T4 GPU access, longer sessions)

---

## 3. Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| PDF Parser | `marker-pdf` | Deep-learning, handles image-heavy slides. Downloads ~4GB models on first run. |
| OCR Fallback | `pytesseract` + `pymupdf` | For slides Marker can't parse (image-only pages) |
| NER / Keyphrase | `urchade/gliner_large-v2.1` | Zero-shot, 6 academic entity labels |
| Sentence Embeddings | `all-MiniLM-L6-v2` (dedup) + `all-mpnet-base-v2` (evidence) | Both from sentence-transformers |
| LLM | Groq → `llama-3.3-70b-versatile` | Free tier works. Fallback: `llama-3.1-8b-instant` |
| LLM Framework | `langchain-groq` | |
| NLP | `spacy en_core_web_sm` | Sentence segmentation only |

**API Keys needed:**
- **Groq API key** — free at console.groq.com (needed for Step 4 only)

---

## 4. Key Config Values (currently in notebook)

```python
# Step 1
PAGE_SEP = "\n\n<<<MARKER_PAGE_BREAK>>>\n\n"

# Step 2
# (no config constants — all logic in code)

# Step 3
GLINER_MODEL        = "urchade/gliner_large-v2.1"
GLINER_THRESHOLD    = 0.35
DEDUP_SIM_THRESHOLD = 0.85
KEYPHRASE_MAX       = 25
HEADING_BOOST       = 0.15

GLINER_LABELS = [
    "Academic Concept", "Theoretical Principle", "Technical Term",
    "Process or Method", "System or Framework", "Formula or Equation",
]

# Step 4
LLM_PRIMARY   = "llama-3.3-70b-versatile"
LLM_FALLBACK  = "llama-3.1-8b-instant"
TRIPLE_CONFIDENCE_THRESHOLD   = 0.70
EVIDENCE_SIMILARITY_THRESHOLD = 0.65
```

---

## 5. Relation Types (Step 4)

The LLM is constrained to exactly 8 relation types with strict direction rules:

| Relation | Direction |
|----------|-----------|
| `isPrerequisiteOf` | subject must be understood BEFORE object |
| `isDefinedAs` | subject IS the concept; object IS the definition |
| `isExampleOf` | subject is a SPECIFIC INSTANCE of object |
| `contrastedWith` | subject AND object are explicitly compared |
| `appliedIn` | subject concept IS USED IN object |
| `isPartOf` | subject is a STRUCTURAL COMPONENT of object |
| `causeOf` | subject DIRECTLY CAUSES object |
| `isGeneralizationOf` | subject is BROADER CATEGORY containing object |

---

## 6. Accuracy Baseline (CS1050-L03 — Java Collections lecture, 22 slides)

This was the first test PDF used for ground-truth evaluation.

| Metric | Value |
|--------|-------|
| Step 1 accuracy (before fixes) | ~80% |
| Step 2 accuracy (before fixes) | ~65% |
| Step 4 triples — Strict accuracy | **50%** (12/24 validated) |
| Step 4 triples — Lenient accuracy | **54.2%** |
| Target after fixes | ~62–65% |

The GLiNER notebook (`Edu_kg_gliner.ipynb`) has all fixes applied but **has NOT been re-run yet on CS1050-L03 to get post-fix accuracy**. That is the first thing to do in the new account.

---

## 7. Test PDFs Used

| File | Description | Status |
|------|-------------|--------|
| `CS1050-L03` | Java Collections lecture, 22 slides, University of Westminster | Accuracy baseline established. All 4 steps run. Pre-fix accuracy: 50% strict. |
| `5COSC001W OOP Week 5` | OOP Arrays/Collections/LinkedList lecture, 28 slides, University of Westminster, Dr. Barbara Villarini | Steps 1 & 2 analyzed and fixed. Steps 3 & 4 NOT YET RUN on this PDF. |

---

## 8. All Fixes Applied So Far

### Round 1 (applied to original notebook → `Edu_kg_fixed.ipynb`)
- **Step 2:** Beamer/LaTeX Unicode bullet markers (◮ ▪ ►) now parsed as bullets
- **Step 2:** `{N}` slide number tags (e.g. `{11}`, `{22}`) added to noise patterns
- **Step 3:** Code density threshold lowered: `0.10 → 0.08`
- **Step 4:** Evidence similarity threshold lowered: `0.75 → 0.65`
- **Step 4:** New system prompt with explicit direction rules, diversity requirements, slide heading rule

### Round 2 (GLiNER replacement → `Edu_kg_gliner.ipynb` — CURRENT NOTEBOOK)
- **Step 3:** Replaced SIFRank/spaCy noun chunks entirely with **GLiNER large-v2.1**
- Threshold: 0.35, 6 universal academic entity labels, heading boost preserved

### Round 3 (latest — `fixed_steps_1_2.py` — ready to paste into notebook)
These fixes were developed in the last session and validated against real data but **not yet merged into the Colab notebook**:

#### Step 1 fixes:
- **FIX S1-1:** Duplicate page detection — Jaccard similarity on word sets between new page and last 2 slides. **Threshold must be 0.50** (not 0.55 — confirmed by real data showing Jaccard = 0.520 for the title page duplicate that was just analysed)
- **FIX S1-2:** `_is_empty_page` now strips `{N}` tags before length check

#### Step 2 fixes:
- **FIX S2-1 (CRITICAL):** Copyright noise pattern `^[cC].*` → `^(?:©|\(c\)|copyright\b)`. The old pattern killed EVERYTHING starting with C/c, including "Collections", "Comparable in Java", "comparison-based sorting"
- **FIX S2-2:** `_is_code_line` func-call rule: `n >= 2` → `n >= 3 and word_count <= 8` (stops "new Set( )" and "new Map( )" being classified as code)
- **FIX S2-3:** Removed `<` and `>` from `_CODE_CHARS` (prevents inequality lists like `<, >, compareTo` being classified as code)
- **FIX S2-4:** camelCase rule now requires `t.count(",") < 2` (prevents comma-separated token lists triggering it)
- **FIX S2-5:** Mixed code/prose fence handling — `_is_prose_label()` function extracts bullet labels from inside ` ``` ` fences (fixes slide_018 losing "construction", "storing a value" etc.)
- **FIX S2-6:** Dot-notation rule added: `identifier.property` on ≤3-word lines → `code_lines` (fixes `scores.length`)

#### TWO MORE FIXES STILL NEEDED (found in latest audit session, not yet coded):
- **FIX S1-1 threshold:** Change dedup threshold from `0.55 → 0.50` in the code (the file says 0.55 but real data shows it needs 0.50)
- **FIX S2 OCR routing:** `'> OCR: ...'` lines match bullet regex `r'^[-*¢—>]\s+'` because `>` is in the character class. The bullet `elif` fires before the blockquote `elif s.startswith('>')` is reached, so OCR lines with `> OCR:` prefix are dumped into bullets with "OCR: " prefix intact. Fix: swap the order — put `elif s.startswith('>')` BEFORE `elif re.match(r'^[-*¢—>]\s+', s)` in `_parse_slide`.

---

## 9. Current Step 2 Accuracy After Round 3 Fixes (OOP Week 5 PDF)

| Category | Count | Slides |
|----------|-------|--------|
| Perfect | 20 | 003–004, 006–014, 016, 018–023, 025, 028 |
| Minor (tolerable) | 5 | 012, 015, 024, 026, 027 |
| Issue | 2 | 005 (Step1 limitation), 017 (image heading) |
| Broken | 1 | 001 (OCR routing bug — fix described above) |
| **Total** | **28** | **85% clean** |

---

## 10. File Inventory

### Notebook files (in Google Drive / download from old account):
| File | Description |
|------|-------------|
| `Edu_kg.ipynb` | Original notebook (all 4 steps, original code) |
| `Edu_kg_fixed.ipynb` | Round 1 fixes applied |
| `Edu_kg_gliner.ipynb` | **CURRENT — Round 2 (GLiNER) + all prior fixes** |

### JSON output files (intermediate results):
| File | Step | PDF | Notes |
|------|------|-----|-------|
| `CS1050-L03_step1_parsed.json` | 1 | CS1050-L03 | |
| `CS1050-L03_step2_preprocessed.json` | 2 | CS1050-L03 | |
| `CS1050-L03_step3_keyphrases.json` | 3 | CS1050-L03 | SIFRank version (old) |
| `CS1050-L03_step4_triples.json` | 4 | CS1050-L03 | 50% strict accuracy (pre-fix baseline) |
| `test3_step1_parsed.json` | 1 | OOP Week 5 | Generated by fixed Step 1 code |
| `test3_step2_preprocessed.json` | 2 | OOP Week 5 | Generated by fixed Step 2 code |

### Code file from last session:
| File | Description |
|------|-------------|
| `fixed_steps_1_2.py` | Complete fixed Step 1 + Step 2 cells, ready to paste into Colab notebook. BUT still needs the two additional fixes from section 8 above (dedup threshold 0.55→0.50, and OCR routing branch order swap). |

---

## 11. Exact Next Steps (in order)

### Step A — Merge Round 3 fixes into the Colab notebook
1. Open `Edu_kg_gliner.ipynb` in Colab
2. Replace the Step 1 cell with the Step 1 code from `fixed_steps_1_2.py`
3. Replace the Step 2 cell with the Step 2 code from `fixed_steps_1_2.py`
4. Apply the two remaining fixes not yet in the file:
   - In `_is_duplicate_page`: change `threshold: float = 0.55` → `threshold: float = 0.50`
   - In `_parse_slide`: move `elif s.startswith(">"):` block to appear BEFORE `elif re.match(r'^[-*¢—>]\s+', s):` block

### Step B — Re-run full pipeline on CS1050-L03
Run all 4 steps on `CS1050-L03` PDF with the fully fixed notebook. Compare Step 4 triple accuracy against the 50% strict / 54.2% lenient baseline. Target: ~62–65%.

### Step C — Run full pipeline on OOP Week 5 PDF
Run all 4 steps on `5COSC001W OOP Week 5` PDF. Steps 3 and 4 have never been run on this PDF. Collect triple accuracy numbers for benchmarking.

### Step D — Paper writing
Use accuracy numbers from both test PDFs for the evaluation section. Compare against Ain et al. 2025 baseline paper.

---

## 12. How to Onboard Claude in the New Account

Paste this exact prompt to start the new session:

---

> I'm continuing a research project called PACE-KG (Pedagogically-Aware, Citation-Evidenced Knowledge Graph). This is a pipeline that converts lecture slide PDFs into knowledge graph triples using Marker PDF parsing, GLiNER NER, and Groq LLM validation.
>
> The full project context is in this handover document — please read it carefully before we do anything else. I'll attach: (1) this handover document, (2) the current notebook `Edu_kg_gliner.ipynb`, and (3) the `fixed_steps_1_2.py` file containing the Step 1 and Step 2 fixes.
>
> The immediate task is to merge the fixes from `fixed_steps_1_2.py` into the notebook, apply two additional small fixes described in section 8 of the handover (dedup threshold 0.50, OCR branch order swap), then run the full pipeline on CS1050-L03 to get post-fix accuracy numbers.

---

## 13. Known Limitations / Future Work

- **slide_005 (OOP deck):** Marker renders `**Arrays**` as bold body text instead of `# **Arrays**` heading because the slide has a large figure on top. Step 2 cannot fix Step 1 classification errors. Mitigation: bold-only text that is short (≤ 3 words) and isolated could be promoted to heading.
- **slide_017 (OOP deck):** Pure continuation table slide — no heading in PDF, so Step 2 correctly outputs no heading. No fix possible.
- **GLiNER accuracy on CS1050-L03:** Not yet measured. The GLiNER notebook was built but never run on the original test PDF. This is the most important unknown.
- **Evaluation set:** Currently only 2 PDFs tested. Need more diverse PDFs (different subjects, different slide styles) for robust benchmarking.
- **Step 4 isPartOf overuse:** System prompt includes a cap at 40% but LLM sometimes still overuses it. May need stricter enforcement.
