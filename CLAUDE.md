# CLAUDE.md — PACE-KG Project

> This file is the single source of truth for AI coding assistants (Claude, Copilot).
> Read this ENTIRE file before writing any code. Do not skip sections.

---

## 1. Project Overview

**PACE-KG** (Pedagogically-Aware, Citation-Evidenced Knowledge Graph) automatically
constructs Educational Knowledge Graphs (EduKGs) from PDF lecture slides.

It is an optimized redesign of the pipeline by Ain et al. (2025):
> arXiv:2509.05392 — "An Optimized Pipeline for Automatic Educational Knowledge Graph Construction"

### Current Implementation Status

The pipeline is implemented as a single self-contained Colab notebook: **`pace_kg.py`**.
The scope for now is: **upload a PDF → run the full 8-step pipeline → store KG in Neo4j AuraDB**.

There is no web app, no FastAPI server, no Celery worker, and no frontend yet.
All of those come later. Do not build them unless explicitly asked.

### What is kept from the original paper
- SIFRankSqueezeBERT-style keyphrase extraction (Step 3, now replaced by GLiNER)
- SBERT all-mpnet-base-v2 for embeddings (Steps 4, 5, 6)
- Neo4j as graph database (Step 7)
- Incremental slide-by-slide storage (available to learners per slide)

### What is replaced / novel in this implementation
| Original | PACE-KG replacement | Step |
|---|---|---|
| PDFMiner | **Marker** (deep-learning PDF parser) with Tesseract OCR fallback | 1 |
| — | **Markdown Preprocessor** (new) | 2 |
| SIFRankSqueezeBERT | **GLiNER large-v2.1** (zero-shot NER for keyphrases) | 3 |
| DBpedia Spotlight | **LLM Triple Extraction** via Groq/Llama-3 | 4 — CORE NOVEL |
| Wikipedia-dependent weighting | **3-Signal internal SBERT weighting** | 5 |
| Wikipedia dump + SPARQL | **Closed-Corpus Expansion** | 6 — NOVEL |

---

## 2. Current File Structure

```
pace-kg/
├── CLAUDE.md          ← this file
├── pace_kg.py         ← THE entire pipeline (Colab notebook exported as .py)
└── (all output JSONs written to Colab runtime / Google Drive)
```

All pipeline logic lives inside `pace_kg.py`. It is structured as sequential cells
matching the 8 pipeline steps. Each step saves a JSON output file and optionally
downloads it to the user's browser.

---

## 3. Runtime Environment

The pipeline runs on **Google Colab** (free T4 GPU recommended for Marker + GLiNER).

### Dependencies installed at runtime (in pace_kg.py Cell 0)
```bash
pip install marker-pdf -q
pip install pytesseract pillow pymupdf -q
apt-get install -y tesseract-ocr -q
pip install sentence-transformers spacy -q
python -m spacy download en_core_web_sm -q
pip install gliner -q
pip install langchain langchain-groq -q
pip install neo4j -q
```

### External services required
| Service | Purpose | How to get |
|---|---|---|
| Groq API | LLM calls in Steps 4 and 6 | Free at console.groq.com |
| Neo4j AuraDB | Graph storage in Steps 7 and 8 | Free tier at neo4j.com/aura |

---

## 4. Pipeline Steps — What Each Step Does

---

### STEP 1 — Marker PDF Parsing

**Input:** PDF file (uploaded via `google.colab.files.upload()`)
**Output:** `{STEM}_step1_parsed.json` — list of `SlideMarkdown` objects

**How it works:**
- Runs Marker (deep-learning PDF parser) with `paginate_output=True` using a custom `PAGE_SEP` string
- Splits full markdown output on `PAGE_SEP` to get per-page markdown
- For pages where Marker returns no text (`_is_empty_page()`), falls back to Tesseract OCR at 3× zoom
- OCR-recovered lines are prefixed `> OCR:` so Step 2 can re-classify them
- Duplicate pages are detected by Jaccard similarity on word sets (threshold 0.50, window 2) and skipped

**Key data model:**
```python
@dataclass
class SlideMarkdown:
    slide_id:     str   # e.g. "slide_003"
    page_number:  int   # 1-indexed
    raw_markdown: str
    doc_id:       str   # = Path(PDF_PATH).stem
```

**Important fixes in current code:**
- `_is_empty_page()` strips `{N}` slide-number tags before the length check (FIX S1-2)
- Duplicate detection uses Jaccard on word sets, not string equality (FIX S1-1)

---

### STEP 2 — Markdown Preprocessor

**Input:** `slides_md` (list of SlideMarkdown from Step 1)
**Output:** `{STEM}_step2_preprocessed.json` — list of `SlideContent` objects

**Four stages:**

1. **Structural parsing** — routes each line into typed buckets:
   - `#`-prefixed lines → `headings` + `heading_phrases`
   - `- * ¢ — >` prefixed lines → `bullets`
   - `| cell |` lines → `table_cells`
   - `> OCR:` blockquotes → re-classified by `_classify_ocr_line()`
   - Other `>` blockquotes → `captions`
   - Everything else → `body_text`
   - Lines inside ` ``` ` fences → `code_lines` (prose bullet labels inside fences go to `bullets` — FIX S2-5)

2. **Noise removal** — lines matching `_NOISE_PATTERNS` are discarded:
   - Page numbers, bare digits, copyright notices, bibliography headers
   - URLs, footnote references `[N]`, image markdown, quiz labels
   - `{N}` slide-number tags, infrastructure/legend labels

3. **Cross-slide repetition filter** — blocks appearing on >50% of slides are removed from all slides (catches recurring headers/footers). Requires ≥5 slides to run.

4. **Assembly** — `clean_text = " ".join(headings + body_text + bullets + table_cells + captions)`

**Key fixes in current code:**
- Copyright pattern only matches actual copyright lines, not any text starting with C (FIX S2-1)
- `_is_code_line()` excludes `<>` from code chars to avoid false positives on inequality lists (FIX S2-3)
- Code line: func-call rule requires n≥3 AND word_count≤8 (FIX S2-2)
- camelCase rule requires <2 commas (FIX S2-4)
- New dot-access rule for short `identifier.property` expressions (FIX S2-6)
- Prose bullet labels inside code fences go to bullets not code_lines (FIX S2-5)

---

### STEP 3 — Keyphrase Extraction (GLiNER)

**Input:** `slide_contents` (list of SlideContent from Step 2)
**Output:** `{STEM}_step3_keyphrases.json` — dict of `slide_id → List[Keyphrase]`

**Models used:**
- `urchade/gliner_large-v2.1` — zero-shot NER for academic/technical concepts
- `all-MiniLM-L6-v2` — SBERT for near-duplicate deduplication

**GLiNER entity labels** (generalized for any academic domain):
```python
GLINER_LABELS = [
    "Academic Concept",
    "Theoretical Principle",
    "Technical Term",
    "Process or Method",
    "System or Framework",
    "Formula or Equation",
]
```

**Extraction pipeline per slide:**
1. Build `extract_text` = headings first + rest (code_lines excluded)
2. Run `gliner.predict_entities()` with threshold 0.35
3. Collapse duplicate spans — keep highest score per phrase
4. Drop phrases shorter than 3 chars
5. Assign `source_type` by checking which bucket contains the phrase (headings → bullets → table → caption → body)
6. Apply heading boost: `score = min(score + 0.15, 1.0)` if source_type == "heading"
7. Deduplicate near-synonyms via SBERT cosine ≥ 0.85
8. Sort by score descending, cap at 25 per slide

**Pedagogical filter:** Title slides (page 1, <40 words) and near-empty slides (<8 words) are skipped.

**Key data model:**
```python
@dataclass
class Keyphrase:
    phrase:      str    # lowercase
    score:       float
    source_type: str    # heading | bullet | table | caption | body
    slide_id:    str
    doc_id:      str
    appears_in:  str    # sentence containing the phrase
```

---

### STEP 4 — LLM Triple Extraction ← CORE NOVEL CONTRIBUTION

**Input:** keyphrases per slide + SlideContent
**Output:** `{STEM}_step4_triples.json` — list of `Triple` objects

**LLM:** Groq `llama-3.3-70b-versatile` (primary), `llama-3.1-8b-instant` (fallback on HTTP 429)
Both use `temperature=0` and `response_format: {type: json_object}`.

**8 relation types with direction rules:**

| Relation | Direction rule |
|---|---|
| `isPrerequisiteOf` | subject must be understood BEFORE object |
| `isDefinedAs` | subject IS the concept; object is the definition |
| `isExampleOf` | subject is a SPECIFIC INSTANCE of broader object |
| `contrastedWith` | subject AND object explicitly compared |
| `appliedIn` | subject concept IS USED IN / runs INSIDE object |
| `isPartOf` | subject is a PHYSICAL/STRUCTURAL COMPONENT of object |
| `causeOf` | subject DIRECTLY CAUSES object |
| `isGeneralizationOf` | subject is BROADER CATEGORY containing object |

**Diversity requirement (in prompt):**
- `isPartOf` capped at 40% of triples in any response
- Must produce `contrastedWith` when slide title contains "vs"/"comparison"/"advantages"
- Slide title must NOT be used as triple object (heading rule)

**3-layer validation (all three must pass):**

| Layer | Check |
|---|---|
| 1 — Anchor | subject AND object must be in the keyphrase set (lowercase); relation must be one of 8 types; subject ≠ object |
| 2 — Evidence | SBERT cosine between evidence string and slide clean_text ≥ 0.65 (uses `all-mpnet-base-v2`) |
| 3 — Confidence | LLM-reported confidence ≥ 0.70 |

**Key data model:**
```python
@dataclass
class Triple:
    subject:    str
    relation:   str
    object:     str
    evidence:   str    # exact sentence from slide — never paraphrased
    confidence: float
    slide_id:   str
    doc_id:     str
    source:     str = "extraction"
```

---

### STEP 5 — Concept Weighting & Pruning

**Input:** `{STEM}_step3_keyphrases.json`, `{STEM}_step4_triples.json`, `{STEM}_step2_preprocessed.json`
**Output:** `{STEM}_step5_concepts.json`, `{STEM}_step5_triples_pruned.json`

**Weight formula:**
```
final_weight = (0.5 × w_evidence) + (0.3 × w_slide) + (0.2 × w_doc)
               + relation_role_boost + source_type_boost
```
- `w_evidence` — SBERT cosine between concept phrase and its best evidence sentence from Step 4 triples
- `w_slide` — SBERT cosine between concept phrase and all slides it appears on (concatenated)
- `w_doc` — SBERT cosine between concept phrase and full document text
- All three signals use `all-mpnet-base-v2`

**Relation role boosts:**
```python
RELATION_ROLE_BOOSTS = {
    ("isDefinedAs",        "object"):  +0.15,
    ("isPrerequisiteOf",   "subject"): +0.10,
    ("isGeneralizationOf", "object"):  +0.10,
    ("contrastedWith",     "subject"): +0.05,
    ("contrastedWith",     "object"):  +0.05,
    ("causeOf",            "subject"): +0.05,
    ("isExampleOf",        "subject"): -0.05,
}
```

**Source type boosts:** heading +0.10, bullet +0.05, table +0.05, body 0.00, caption -0.05

**Pruning threshold:** `WEIGHT_THRESHOLD = 0.192` — concepts below this are dropped.

**Semantic merge:** Concepts with SBERT cosine ≥ 0.92 are merged (lower-weight into higher-weight).
Concepts with cosine ≥ 0.75 but < 0.92 are flagged `needs_review = True`.

**Key data model:**
```python
@dataclass
class ConceptNode:
    name:            str
    aliases:         List[str]
    slide_ids:       List[str]
    source_type:     str
    keyphrase_score: float
    final_weight:    float
    doc_id:          str
    needs_review:    bool = False
```

---

### STEP 6 — Closed-Corpus Concept Expansion

**Input:** `{STEM}_step2_preprocessed.json`, `{STEM}_step3_keyphrases.json`, `{STEM}_step5_triples_pruned.json`
**Output:** `{STEM}_step6_expansion.json` — list of `ExpansionEdge` objects

**Key constraint: NO external knowledge.** Only concepts from the document itself are used.

**Document vocabulary** is built from:
- All keyphrases from Step 3
- All triple subjects/objects from Step 5
- spaCy noun chunks from all slide clean_text (articles stripped, e.g. "a list" → "list")

**4-phase expansion per core concept:**
1. **LLM selection** — ask LLM to select related concepts from a pool of up to 80 vocabulary items (pool excludes already-linked concepts)
2. **SBERT gate** — keep only candidates with cosine ≥ 0.65 vs core concept
3. **Slide-scope constraint** — keep only candidates that are: (a) on an adjacent slide (±1 index), OR (b) have SBERT cosine ≥ 0.70 vs full document
4. **Deduplication** — skip self-loops, trivial article-prefix pairs (`"list"` ↔ `"a list"`), already-existing pairs

**Edge relation type:** `relatedConcept` (source = "expansion")

**Key data model:**
```python
@dataclass
class ExpansionEdge:
    subject:    str
    relation:   str    # always "relatedConcept"
    object:     str
    source:     str    # always "expansion"
    confidence: float  # SBERT cosine sim
    slide_id:   str
    doc_id:     str
```

---

### STEP 7 — Neo4j Slide-EduKG Storage

**Input:** All Step 5 and Step 6 outputs
**Output:** Graph stored in Neo4j AuraDB + `{STEM}_step7_storage_report.json`

**Node type:** `Concept`
**Edge type:** `RELATION` (relation_type property distinguishes the 8 types + relatedConcept)
**Also creates:** `LearningMaterial` node with `BELONGS_TO` edges from every concept

**Connection config (hardcoded in pace_kg.py — change per run):**
```python
NEO4J_URI      = "neo4j+s://XXXXXXXX.databases.neo4j.io"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_auradb_password"
```

**Two-pass write strategy:**
- Pass 1: store ALL concept nodes first (nodes must exist before edges reference them)
- Pass 2: store edges slide-by-slide (extraction triples + expansion edges)
- Pass 3: create LearningMaterial node + BELONGS_TO links

**Key Cypher patterns:**
```cypher
-- Node upsert (always MERGE, never CREATE)
MERGE (c:Concept {name: $name})
SET c.aliases = $aliases, c.slide_ids = $slide_ids, ...

-- Edge upsert
MATCH (s:Concept {name: $subject}), (o:Concept {name: $object})
MERGE (s)-[r:RELATION {relation_type: $relation_type, slide_id: $slide_id, doc_id: $doc_id}]->(o)
SET r.evidence = $evidence, r.confidence = $confidence, r.source = $source
```

**Semantic conflict resolution at write time:**
- Incoming concept name is checked against existing `concept_embs` dict
- If cosine ≥ 0.92, the canonical name is used instead (merge)

---

### STEP 8 — LM-EduKG Aggregation

**Input:** Neo4j graph + all intermediate JSONs
**Output:** `{STEM}_step8_lm_edkg.json`, `{STEM}_step8_summary.json`

**Verifies four merge constraints:**
1. Every slide has ≥1 concept (checked via Neo4j query grouping by slide_id)
2. All local concepts exist in Neo4j (set difference check)
3. All extraction edges have non-empty evidence (Neo4j count query)
4. Cross-slide expansion edges are tagged `material_level = true`

**Exports:**
- Full graph (all concept nodes + all edges) as `lm_edkg["nodes"]` and `lm_edkg["edges"]`
- `srs_pool` — extraction-only triples for SRS accuracy evaluation

---

## 5. Data Flow Summary

```
PDF file
  │
  ▼ Step 1 (Marker + OCR fallback)
{STEM}_step1_parsed.json         ← SlideMarkdown[]
  │
  ▼ Step 2 (Markdown Preprocessor)
{STEM}_step2_preprocessed.json  ← SlideContent[]
  │
  ▼ Step 3 (GLiNER keyphrase extraction)
{STEM}_step3_keyphrases.json    ← {slide_id: Keyphrase[]}
  │
  ▼ Step 4 (LLM triple extraction + 3-layer validation)
{STEM}_step4_triples.json       ← Triple[]
  │
  ▼ Step 5 (SBERT weighting + pruning)
{STEM}_step5_concepts.json      ← ConceptNode[]
{STEM}_step5_triples_pruned.json
  │
  ▼ Step 6 (Closed-corpus expansion)
{STEM}_step6_expansion.json     ← ExpansionEdge[]
  │
  ▼ Step 7 (Neo4j storage)
[Neo4j AuraDB graph]
{STEM}_step7_storage_report.json
  │
  ▼ Step 8 (LM-EduKG aggregation)
{STEM}_step8_lm_edkg.json
{STEM}_step8_summary.json
```

---

## 6. Key Configuration Constants

All thresholds are defined inline in `pace_kg.py`. When changing them, search for the constant name.

| Constant | Value | Step | Purpose |
|---|---|---|---|
| `PAGE_SEP` | `\n\n<<<MARKER_PAGE_BREAK>>>\n\n` | 1 | Splits Marker output into pages |
| `KEYPHRASE_MAX` | 25 | 3 | Max keyphrases per slide |
| `GLINER_THRESHOLD` | 0.35 | 3 | GLiNER entity confidence floor |
| `DEDUP_SIM_THRESHOLD` | 0.85 | 3 | SBERT cosine for keyphrase dedup |
| `HEADING_BOOST` | 0.15 | 3 | Score bonus for heading-sourced keyphrases |
| `TRIPLE_CONFIDENCE_THRESHOLD` | 0.70 | 4 | Min LLM confidence to keep a triple |
| `EVIDENCE_SIMILARITY_THRESHOLD` | 0.65 | 4 | Min SBERT cosine for evidence check |
| `WEIGHT_THRESHOLD` | 0.192 | 5 | Prune concepts below this weight |
| `MERGE_SIM_THRESHOLD` | 0.92 | 5, 7 | Auto-merge near-duplicate concepts |
| `REVIEW_SIM_THRESHOLD` | 0.75 | 5 | Flag concepts for human review |
| `SBERT_SIM_THRESHOLD` | 0.65 | 6 | SBERT gate for expansion candidates |
| `DOC_WEIGHT_THRESHOLD` | 0.70 | 6 | High-doc-relevance bypass for scope filter |
| `MAX_CANDIDATES_PER_CONCEPT` | 10 | 6 | Max expansion edges per concept |

---

## 7. STEM Variable

`STEM` is derived from the uploaded PDF filename:
```python
STEM = Path(PDF_PATH).stem   # e.g. "lecture5" if file is "lecture5.pdf"
```

All output files use `STEM` as prefix. In Steps 5-8, `STEM` is hardcoded (currently `"test3"`) and must be updated manually to match the uploaded PDF. This is a known limitation to fix.

---

## 8. Rules for AI Assistants — Read Before Writing Any Code

1. **This is a Colab notebook, not a web app.** Do not add FastAPI, Celery, Redis, or Docker unless explicitly asked. All code runs sequentially in Colab cells.

2. **NEVER instantiate SBERT inside a loop.** Models are loaded once at the top of each step cell. Adding a second load inside a loop adds 30+ seconds per slide.

3. **GLiNER replaced SIFRank.** The old CLAUDE.md mentioned SIFRankSqueezeBERT — that is no longer used. Step 3 uses `GLiNER.from_pretrained("urchade/gliner_large-v2.1")`.

4. **STEM must match across all steps.** Steps 5-8 read JSON files using the `STEM` variable. If the user uploaded a new PDF, `STEM` must be updated in Steps 5-8 cells.

5. **Use MERGE, never CREATE in Neo4j.** Same concept appears across multiple slides. CREATE produces duplicates that break the graph.

6. **All concept names are lowercase and stripped** before validation, storage, and comparison. Enforce at Step 4 input and Step 7 write time.

7. **LLM temperature = 0 always.** Use `ChatGroq`. If call returns HTTP 429 (rate limit), automatically retry with the fallback model (`llama-3.1-8b-instant`). Free tier limit is ~30 requests/minute.

8. **Never skip any of the 3 validation layers in Step 4.** All three must pass (anchor, evidence SBERT, confidence).

9. **Evidence sentences are exact quotes.** The SBERT check in Layer 2 catches fabricated sentences. Never paraphrase evidence.

10. **Step 6 runs AFTER all keyphrases and triples are collected.** The Document Vocabulary requires the full document to be processed first.

11. **Do not use Wikipedia, DBpedia, or any external knowledge base** at any step. Closed-corpus constraint is a core research requirement.

12. **The `needs_review` flag in Step 5** is informational only — flagged concepts are still kept in the graph. Only concepts below `WEIGHT_THRESHOLD` are pruned.

13. **OCR fallback lines are prefixed `> OCR:`** in Step 1 output. Step 2's `_classify_ocr_line()` handles re-classification. Do not strip this prefix before Step 2 runs.

---

## 9. Known Limitations / Things to Fix

- `STEM` is hardcoded in Steps 5-8 as `"test3"` — should be passed as a variable from the upload cell
- Neo4j credentials are hardcoded in Steps 7-8 — should read from a Colab secret or env var
- Step 7 `run_query` uses `session.execute_write` (single-item transactions) — batching would be faster
- Groq API key is requested via `getpass` in Steps 4 and 6 separately — should be requested once and stored in `os.environ`
- No progress bar in Steps 5-6 for long PDFs

---

## 10. Research Context

This project targets academic publication comparing against:
- **Baseline:** Ain et al. (2025) optimized pipeline — accuracy 0.47
- **Target:** PACE-KG accuracy > 0.47 using same SRS evaluation method

Two research claims must be preserved in every implementation decision:
1. **Evidence anchoring** — every extraction triple must have a source sentence from the PDF
2. **Closed-corpus expansion** — no concept from outside the PDF enters the KG

The evaluation PDF, expert annotators, and SRS sampling method must match
the original paper exactly. Evaluation code must be kept completely separate from
pipeline code.

The `srs_pool` field in `{STEM}_step8_lm_edkg.json` contains the extraction triples
to use for SRS evaluation sampling.