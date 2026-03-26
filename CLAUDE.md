# CLAUDE.md — PACE-KG Project

> This file is the single source of truth for AI coding assistants (Claude, Copilot).
> Read this ENTIRE file before writing any code. Do not skip sections.
>
> **Reference implementation**: `edu-kg glinger.py` in the repo root is the main working Colab version. 
> Use it as the primary guide, especially for Steps 1–4.

---

## 1. Project Overview

**PACE-KG** (Pedagogically-Aware, Citation-Evidenced Knowledge Graph) automatically
constructs Educational Knowledge Graphs (EduKGs) from PDF lecture slides.

It is an optimized redesign of the pipeline by Ain et al. (2025):
> arXiv:2509.05392 — "An Optimized Pipeline for Automatic Educational Knowledge Graph Construction"

### Current Architecture

The system has two parts:

1. **Pipeline** (`edu-kg glinger.py`) — The complete reference implementation that was developed 
   and tested in Google Colab. This processes a PDF through all 9 steps (Steps 1-8 + Step 9 summaries). 
   **Use this as the primary reference for all implementation.**

2. **Web application** — A production system consisting of:
   - **Backend**: FastAPI application running on Vast.ai RTX 3090 GPU that executes the full pipeline
   - **Frontend**: React application running locally that connects to the backend via REST API

The frontend communicates with the **FastAPI backend** that runs the pipeline and serves the results. 
The backend exposes:
- `POST /upload` — accepts a PDF, runs the full 9-step pipeline, returns `doc_id`
- `GET /status/{doc_id}` — returns current pipeline progress and step
- `GET /graph/{doc_id}` — returns nodes and edges for graph rendering
- `GET /summaries/{doc_id}` — returns per-slide summaries as JSON
- `GET /export/{doc_id}` — returns a downloadable Word document of the summaries

### What the user sees in the frontend
1. Upload a PDF via a drag-and-drop or file picker
2. A progress indicator while the pipeline runs (Steps 1–8 + Step 9 summaries)
3. An **interactive knowledge graph** — nodes are concepts, edges are relations,
   clicking a node shows the source slide and evidence sentence
4. A **slide-by-slide summary panel** — each slide's heading, 2–3 sentence
   summary, and key terms listed below the graph

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
| SIFRankSqueezeBERT | **GLiNER large-v2.1** + LLM fallback (zero-shot NER) | 3 |
| DBpedia Spotlight | **LLM Triple Extraction** via Groq/Llama-3 | 4 — CORE NOVEL |
| Wikipedia-dependent weighting | **3-Signal internal SBERT weighting** | 5 |
| Wikipedia dump + SPARQL | **Closed-Corpus Expansion** | 6 — NOVEL |
| — | **Slide-Level LLM Summaries** | 9 — NEW |

---

## 2. File Structure

```
Edu-KG/
├── CLAUDE.md                    ← this file (in repo root)
├── edu-kg glinger.py            ← THE complete reference pipeline (Colab version)
├── pace-kg/
│   ├── backend/
│   │   ├── main.py              ← FastAPI app (BUILT)
│   │   ├── pipeline_runner.py   ← wraps edu-kg glinger.py steps (BUILT)
│   │   ├── core/
│   │   │   └── config.py        ← Environment-based config (BUILT)
│   │   ├── api/
│   │   │   └── models/
│   │   │       └── schemas.py   ← Pydantic models (BUILT)
│   │   ├── requirements.txt     ← All dependencies (BUILT)
│   │   ├── Dockerfile           ← NVIDIA CUDA base (BUILT)
│   │   ├── .env.example         ← Credentials template (BUILT)
│   │   ├── README.md            ← Backend docs (BUILT)
│   │   ├── QUICKSTART.md        ← Quick start guide (BUILT)
│   │   ├── TODO_STEPS_5-8.md    ← Implementation guide (BUILT)
│   │   └── DEPLOYMENT.md        ← Vast.ai deployment guide (BUILT)
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   ├── components/
│   │   │   │   ├── GraphView.jsx    ← interactive graph (D3.js or vis.js)
│   │   │   │   └── SummaryPanel.jsx ← slide summaries sidebar
│   │   └── package.json
│   └── docker-compose.yml       ← GPU-enabled compose (BUILT)
```

All pipeline logic is extracted from `edu-kg glinger.py`. Steps are sequential functions.
Each step saves a JSON output file using `{STEM}` as the filename prefix.

**`STEM` in Colab vs Backend:**
- **Colab**: `STEM = Path(PDF_PATH).stem` (e.g. "lecture5" if file is "lecture5.pdf")
- **Backend**: `STEM = doc_id` (UUID generated at upload, e.g. "550e8400-e29b-41d4-a716-446655440000")

`STEM` must never be hardcoded. In the backend, it flows from the upload endpoint through every step.

---

## 3. Runtime Environment

**Primary runtime is a GPU-enabled backend host (e.g., Vast.ai RTX 3090) running the full pipeline inside Docker.**

Google Colab was used for development/experimentation only. Production runs entirely on the Vast.ai backend with:
- **Backend**: Runs on Vast.ai RTX 3090 GPU (FastAPI + pipeline)
- **Frontend**: Runs locally, connects to backend via REST API
- **LLM**: All calls via Groq API (no local LLM hosting required)

### Dependencies
```bash
pip install marker-pdf -q
pip install pytesseract pillow pymupdf -q
apt-get install -y tesseract-ocr -q
pip install sentence-transformers spacy -q
python -m spacy download en_core_web_sm -q
pip install gliner -q
pip install langchain langchain-groq -q
pip install neo4j -q
pip install python-docx -q        # for Step 9 Word export
```

### External services required
| Service | Purpose | How to get |
|---|---|---|
| Groq API | LLM calls in Steps 3 (fallback), 4, 6, and 9 | Free at console.groq.com |
| Neo4j AuraDB | Graph storage in Steps 7 and 8 | Free tier at neo4j.com/aura |

### Groq API key — single point of entry
The Groq API key is requested **once** at the start of Step 4 and stored in
`os.environ["GROQ_API_KEY"]`. All subsequent steps read from the environment.
**Do not add another `getpass()` call anywhere.** If a new step needs Groq,
read `os.environ["GROQ_API_KEY"]` directly.

### LLM Model Selection — Use Groq, Not Local Models
**All LLM inference uses Groq API.** Do NOT host models locally on Vast.ai.

**Why Groq (not local)**:
- `llama-3.3-70b-versatile` (70B) cannot run on RTX 3090 (24GB VRAM)
- Groq provides optimized throughput and low latency
- Free tier: ~30 requests/minute (sufficient with batching)
- Auto-fallback to 8B model on rate limits

**Model usage by step**:
- **Step 3 fallback**: `llama-3.1-8b-instant` (cheap, fast keyphrase extraction)
- **Step 4 primary**: `llama-3.3-70b-versatile` (best accuracy for triple extraction)
- **Step 4 fallback**: `llama-3.1-8b-instant` (on HTTP 429)
- **Step 6 expansion**: `llama-3.3-70b-versatile` (concept selection requires 70B quality)
- **Step 9 summaries**: `llama-3.1-8b-instant` (narration task, 8B sufficient)

**Do NOT switch to other models** (e.g., Qwen, Mixtral) unless benchmarked on the pipeline.
The current pairing is optimized for this use case.

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
    page_number:  int   # 1-indexed, matches Marker's output numbering
    raw_markdown: str
    doc_id:       str   # = STEM
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
   - Unicode Beamer markers (◮ ►) → `bullets` (stripped of the marker prefix)
   - `| cell |` lines → `table_cells`
   - `> OCR:` blockquotes → re-classified by `_classify_ocr_line()`
   - Other `>` blockquotes → `captions`
   - Everything else → `body_text`
   - Lines inside ` ``` ` fences → `code_lines` (prose bullet labels inside fences go to `bullets` — FIX S2-5)

2. **Noise removal** — lines matching `_NOISE_PATTERNS` are discarded:
   - Page numbers, bare digits, copyright notices, bibliography headers
   - URLs, footnote references `[N]`, image markdown, quiz labels
   - `{N}` slide-number tags, infrastructure/legend labels

3. **Cross-slide repetition filter** — blocks appearing on >50% of slides are removed
   from all slides (catches recurring headers/footers). Requires ≥5 slides to run.

4. **Assembly** — `clean_text = " ".join(headings + body_text + bullets + table_cells + captions)`

**Key fixes in current code:**
- Copyright pattern only matches actual copyright lines, not any text starting with C (FIX S2-1)
- `_is_code_line()` excludes `<>` from code chars to avoid false positives on inequality lists (FIX S2-3)
- Code line: func-call rule requires n≥3 AND word_count≤8 (FIX S2-2)
- camelCase rule requires <2 commas (FIX S2-4)
- New dot-access rule for short `identifier.property` expressions (FIX S2-6)
- Prose bullet labels inside code fences go to bullets not code_lines (FIX S2-5)

**Key data model:**
```python
@dataclass
class SlideContent:
    slide_id:        str
    page_number:     int
    doc_id:          str
    headings:        List[str]
    body_text:       List[str]
    bullets:         List[str]
    table_cells:     List[str]
    captions:        List[str]
    code_lines:      List[str]
    heading_phrases: List[str]
    clean_text:      str
    ocr_applied:     bool
```

---

### STEP 3 — Keyphrase Extraction (GLiNER + LLM fallback)

**Input:** `slide_contents` (list of SlideContent from Step 2)
**Output:** `{STEM}_step3_keyphrases.json` — dict of `slide_id → List[Keyphrase]`

**Models used:**
- `urchade/gliner_large-v2.1` — primary: zero-shot NER for academic/technical concepts
- `llama-3.1-8b-instant` via Groq — fallback: used when GLiNER returns 0 or 1 phrases on a content slide
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
3. If GLiNER returns ≤1 phrase AND slide has ≥20 words → call LLM fallback
4. LLM fallback: ask `llama-3.1-8b-instant` to extract up to 12 keyphrases as JSON
5. LLM-sourced keyphrases always get `score = 0.40` (fixed) and are identifiable by this score
6. Collapse duplicate spans — keep highest score per phrase
7. Drop phrases shorter than 3 chars
8. Assign `source_type` by checking which bucket contains the phrase
9. Apply heading boost: `score = min(score + 0.15, 1.0)` if source_type == "heading"
10. Deduplicate near-synonyms via SBERT cosine ≥ 0.85
11. Sort by score descending, cap at 25 per slide

**Pedagogical filter:** Title slides (page 1, <40 words) and near-empty slides (<8 words) are skipped.

**`is_llm_slide` flag:** Set to `True` in Step 4 when ALL keyphrases on a slide have
`score == 0.40` (meaning every keyphrase came from the LLM fallback). This flag
selects a lower SBERT evidence threshold (0.50 instead of 0.65) in Step 4 Layer 2,
because LLM-sourced slides typically have terse bullet-list text.

**Key data model:**
```python
@dataclass
class Keyphrase:
    phrase:      str    # lowercase
    score:       float  # GLiNER score (0.35–1.0) or fixed 0.40 for LLM-sourced
    source_type: str    # heading | bullet | table | caption | body | injected
    slide_id:    str
    doc_id:      str
    appears_in:  str    # sentence containing the phrase
```

Note: `source_type = "injected"` is used by Step 4's cross-slide enrichment (see Step 4).
It is a valid value and must not be filtered out.

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

---

#### Cross-slide keyphrase enrichment (runs before validation)

Some slides have keyphrases that are all members of a category (e.g. `dequeue`, `enqueue`)
but the parent concept (`queue`) lives only in the previous slide's keyphrase list. Without
the parent, no valid triples can be formed.

**Mechanism:** At the start of each slide's processing in the extraction loop, the
single highest-scored keyphrase from the immediately preceding slide is injected into
the current slide's `kp_set` and shown to the LLM in the prompt — but only if it is
not already present in the current slide's keyphrases.

```python
injected = replace(best_prev, slide_id=sc.slide_id, score=0.35, source_type="injected")
kps.append(injected)
```

- Injected keyphrases have `source_type = "injected"` and `score = 0.35`
- At most one keyphrase is injected per slide
- The `is_llm_slide` flag excludes injected keyphrases from its calculation:
  `original_kps = [k for k in kps if k.source_type != "injected"]`
  `is_llm_slide = all(k.score == 0.40 for k in original_kps)`

---

#### 4-layer validation (ALL FOUR layers must pass — do not skip any)

| Layer | Check |
|---|---|
| 1 — Anchor | subject AND object in keyphrase set (lowercase, parentheses stripped); relation is one of 8 types; subject ≠ object; neither is a structural label |
| 2 — Evidence | SBERT cosine between evidence string and slide `clean_text` ≥ 0.65 (GLiNER slides) or ≥ 0.50 (LLM slides). Uses `all-mpnet-base-v2` |
| 3 — Confidence | LLM-reported confidence ≥ 0.70 |
| 4 — Semantic post-validation | 5 rules described below — catches direction errors and factual mistakes that prompting alone cannot fix |

**Layer 4 rules in detail:**

`_in_ev(phrase, ev_lower)` is a word-boundary-safe phrase search helper used by all L4 rules.
It prevents "map" matching inside "hashmap". It also tries plural/singular variants and the
root content word of multi-word phrases. It is defined as a nested function inside `validate_triple`.

**L4-2 — isDefinedAs checks:**
- Evidence must contain a definitional signal word: `" is "`, `" are "`, `" called "`,
  `" refers to "`, `" means "`, `" defined as "`, `" known as "`, `" denoted "`,
  `" represents "`, `" represented as "`, `" can be "`, `" consist "`
- Both subject AND object must appear in the evidence sentence (word-boundary safe)
- If `"called {obj}"` is in evidence but `"called {subj}"` is not → direction is reversed → reject
- If `"can be represented as"` is in evidence → object is one possible form, not a definition → reject

**L4-3 — isPartOf usage signal:**
- If evidence contains `"passed"`, `" used in "`, `"operates in"`, `" works in "`,
  `"applied to"`, `" referenced "`, `" referenced using "`, `" accessed "` → reject
- These signals mean the relationship is passing/usage (appliedIn), not structural membership

**L4-4 — isExampleOf IS-A check:**
- If evidence has an IS-A signal word (`" is a "`, `" is an "`, `" are a "`, `" such as "`,
  `" example "`, `" instance "`, `" implementation "`, `" type of "`, `" kind of "`,
  `" includes "`, `" like "`, `"implementations"`) → allow through regardless of positions
- If NO IS-A signal AND object not in evidence AND object is not an injected keyphrase → reject
- If NO IS-A signal AND object not in evidence AND object IS an injected keyphrase →
  require subject appears in evidence (grounded), then allow
- If NO IS-A signal AND both in evidence → require subject appears before object (ordered)
- If `"making"` appears between subject and object positions → reject
  (object is a predicate "X becomes Y", not a category)

**L4-5 — isGeneralizationOf helper signal:**
- If evidence contains `" helper "`, `" abstract class for "`, `" utility class "`,
  `" base class for "`, `" supports "`, `" assists "` → reject
- These describe implementation-helper relationships, not category membership

---

#### Post-validation deduplication

After all triples for a slide pass Layer 4, `_dedup_triples()` runs two passes:

**Pass 1 — Exact pair dedup:** For any two triples with the same (subject, object) pair,
keep only the one with the highest confidence. Handles cases where LLM generates both
`isPartOf` and `isExampleOf` for the same pair.

**Pass 2 — Semantic pair dedup:** Uses `_phrases_are_synonyms(a, b)` to detect:
1. Acronym/expansion pairs: one string is a short acronym (≤6 alpha chars) and the
   other is its expansion (e.g. "fifo" and "first in, first out" → same concept)
2. Substring containment with word-boundary guard (e.g. "ordered list" vs
   "ordered list of values" → same concept; "array" vs "arraylist" → different)

If two triples have semantically equivalent (subject, object) pairs, keep only the
higher-confidence one. This prevents "queue isDefinedAs fifo" AND
"queue isDefinedAs first in, first out" both surviving.

---

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
The `needs_review` flag is informational only — flagged concepts are still kept.

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
This is a core research requirement. Do not add Wikipedia, DBpedia, or any external KB.

Step 6 runs AFTER all keyphrases and triples are collected. The Document Vocabulary
requires the full document to be processed first.

**Document vocabulary** is built from:
- All keyphrases from Step 3
- All triple subjects/objects from Step 5
- spaCy noun chunks from all slide clean_text (articles stripped, e.g. "a list" → "list")

**4-phase expansion per core concept:**
1. **LLM selection** — ask LLM to select related concepts from pool of up to 80 vocabulary items
2. **SBERT gate** — keep only candidates with cosine ≥ 0.65 vs core concept
3. **Slide-scope constraint** — keep only candidates that are: (a) on an adjacent slide (±1 index),
   OR (b) have SBERT cosine ≥ 0.70 vs full document
4. **Deduplication** — skip self-loops, trivial article-prefix pairs, already-existing pairs

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

**Connection config (read from environment — never hardcode credentials):**
```python
NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
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

### STEP 9 — Slide-Level Summary Generation ← DELIVERABLE

**Input:** `{STEM}_step8_lm_edkg.json` (nodes + edges) + `{STEM}_step2_preprocessed.json`
**Output:** `{STEM}_step9_summaries.json` — list of `SlideSummary` objects

This step narrates the extracted knowledge into human-readable revision notes.
It does NOT re-extract anything — it uses what Steps 1–8 already produced.

**LLM:** `llama-3.1-8b-instant` via Groq, `temperature=0`. Use the cheap model —
each prompt is small (<500 tokens) and this task does not require the large model.
Read the Groq key from `os.environ["GROQ_API_KEY"]`.

**Per-slide logic:**
1. Get the slide's heading from `headings[0]` in the Step 2 output (or `slide_id` if no heading)
2. Get keyphrases for this slide from `lm_edkg["nodes"]` filtered by `slide_ids`
3. Get triples for this slide from `lm_edkg["edges"]` filtered by `slide_id`
4. Build prompt (see below) and call the LLM

**If a slide has zero triples** (diagram-only, code-only, or near-empty slides):
do not call the LLM. Emit: `"This slide contains a diagram — refer to the original slides."`

**Prompt shape:**
```
SLIDE TITLE: {heading}

KEY CONCEPTS: {comma-separated keyphrase names}

RELATIONSHIPS:
- {subject} {relation} {object}
- {subject} {relation} {object}
...

SLIDE TEXT:
{clean_text}

Write a 2-3 sentence summary a student can use as revision notes.
Do not use bullet points. Write in plain English prose.
```

**Key data model:**
```python
@dataclass
class SlideSummary:
    slide_id:    str
    page_number: int
    heading:     str
    summary:     str       # 2-3 sentences or the diagram placeholder
    key_terms:   List[str] # keyphrase names for this slide
    doc_id:      str
```

**Word document export:**
After all summaries are generated, produce `{STEM}_summaries.docx` using `python-docx`.
Structure per slide:
```
[Heading — Bold, 14pt]
[2-3 sentence summary — Normal, 11pt]
Key terms: term1, term2, term3  [Italic, 11pt]
[Page break between slides]
```

---

## 5. Data Flow Summary

```
PDF file
  │
  ▼ Step 1 (Marker + OCR fallback)
{STEM}_step1_parsed.json           ← SlideMarkdown[]
  │
  ▼ Step 2 (Markdown Preprocessor)
{STEM}_step2_preprocessed.json    ← SlideContent[]
  │
  ▼ Step 3 (GLiNER + LLM fallback keyphrase extraction)
{STEM}_step3_keyphrases.json      ← {slide_id: Keyphrase[]}
  │
  ▼ Step 4 (LLM triple extraction + 4-layer validation)
{STEM}_step4_triples.json         ← Triple[]
  │
  ▼ Step 5 (SBERT weighting + pruning)
{STEM}_step5_concepts.json        ← ConceptNode[]
{STEM}_step5_triples_pruned.json
  │
  ▼ Step 6 (Closed-corpus expansion)
{STEM}_step6_expansion.json       ← ExpansionEdge[]
  │
  ▼ Step 7 (Neo4j storage)
[Neo4j AuraDB graph]
{STEM}_step7_storage_report.json
  │
  ▼ Step 8 (LM-EduKG aggregation)
{STEM}_step8_lm_edkg.json         ← full graph export + srs_pool
{STEM}_step8_summary.json
  │
  ▼ Step 9 (Slide-level summaries)
{STEM}_step9_summaries.json       ← SlideSummary[]
{STEM}_summaries.docx             ← Word export for students
  │
  ▼ Frontend
Interactive graph (from lm_edkg nodes/edges)
Slide summary panel (from step9_summaries.json)
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
| `LLM_FALLBACK_SCORE` | 0.40 | 3 | Fixed score assigned to all LLM-sourced keyphrases |
| `TRIPLE_CONFIDENCE_THRESHOLD` | 0.70 | 4 | Min LLM confidence to keep a triple |
| `EVIDENCE_SIMILARITY_THRESHOLD` | 0.65 | 4 | Min SBERT cosine for evidence check (GLiNER slides) |
| `EVIDENCE_SIMILARITY_THRESHOLD_LLM` | 0.50 | 4 | Min SBERT cosine for evidence check (LLM slides) |
| `WEIGHT_THRESHOLD` | 0.192 | 5 | Prune concepts below this weight |
| `MERGE_SIM_THRESHOLD` | 0.92 | 5, 7 | Auto-merge near-duplicate concepts |
| `REVIEW_SIM_THRESHOLD` | 0.75 | 5 | Flag concepts for human review |
| `SBERT_SIM_THRESHOLD` | 0.65 | 6 | SBERT gate for expansion candidates |
| `DOC_WEIGHT_THRESHOLD` | 0.70 | 6 | High-doc-relevance bypass for scope filter |
| `MAX_CANDIDATES_PER_CONCEPT` | 10 | 6 | Max expansion edges per concept |

---

## 7. Frontend — What to Build

The frontend is a single-page React app. It has two main views.

### Upload view
- PDF drag-and-drop or file picker
- On submit: `POST /upload` with the PDF as `multipart/form-data`
- Show a step-by-step progress indicator while the pipeline runs
  (backend streams progress events via Server-Sent Events or WebSocket)
- Steps shown: Parsing → Preprocessing → Keyphrases → Triples → Weighting →
  Expansion → Storing → Aggregating → Generating Summaries

### Results view (shown after pipeline completes)
Two panels side by side:

**Left panel — Interactive Knowledge Graph**
- Render using D3.js force-directed graph or vis.js
- Nodes = concepts (`final_weight` controls node size)
- Edges = relations (colour-coded by relation type)
- Clicking a node shows: source slide, evidence sentence, key terms
- Edge labels show the relation type
- Filter controls: by slide, by relation type, by minimum weight
- Data source: `GET /graph/{doc_id}` → returns `lm_edkg["nodes"]` and `lm_edkg["edges"]`

**Right panel — Slide Summaries**
- Scrollable list of slide cards
- Each card: slide heading (bold), 2-3 sentence summary, key terms as chips
- Clicking a card highlights that slide's concepts in the graph
- A "Download Word document" button at the top
- Data source: `GET /summaries/{doc_id}` → returns `step9_summaries.json`

---

## 8. Backend API — What to Build

**STATUS: Backend is BUILT and ready.** See `pace-kg/backend/` for complete implementation.

FastAPI app in `backend/main.py`. Single-user for now (no auth required).

```python
POST   /upload              # accepts PDF, runs pipeline, returns {doc_id, status}
GET    /status/{doc_id}     # returns current pipeline step and progress
GET    /graph/{doc_id}      # returns {nodes: [...], edges: [...]} from step8_lm_edkg
GET    /summaries/{doc_id}  # returns [{slide_id, heading, summary, key_terms}, ...]
GET    /export/{doc_id}     # returns the .docx file as a download
```

**Implementation status:**
- ✅ All 5 API endpoints implemented
- ✅ Background task worker with progress tracking
- ✅ Steps 1-4 and Step 9 fully implemented
- ⚠️ Steps 5-8 have placeholder implementations (see `TODO_STEPS_5-8.md`)

**Pipeline execution:**
- The pipeline runs as a background task (FastAPI `BackgroundTasks`)
- Each step writes its JSON output to `outputs/{doc_id}/` directory
- Progress is tracked by updating the in-memory job status
- `STEM` for a job = `doc_id` = UUID generated at upload time. Never use the original filename as STEM in production — use the UUID to avoid collisions.

**Deployment architecture:**
- **Backend**: Vast.ai RTX 3090 GPU instance running Docker container
- **Frontend**: Local development machine connecting via REST API
- **LLM**: Groq API (no local hosting)
- **Graph DB**: Neo4j AuraDB (cloud-hosted)

**GPU requirement:**
Marker and GLiNER require a GPU. The backend Docker container must have NVIDIA passthrough.
Use an NVIDIA/CUDA base image (e.g., `nvidia/cuda:12.1.0-base-ubuntu22.04`) to ensure GPU support.

```dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment variables (read from `.env`, never hardcoded):**
```
GROQ_API_KEY=your_key
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## 9. Rules for AI Assistants — Read Before Writing Any Code

1. **STEM is never hardcoded.** It always comes from `Path(PDF_PATH).stem` in Colab,
   or from the job UUID in the web backend. Never write `STEM = "test3"` or any
   literal string. If you see it hardcoded in any step, fix it.

2. **NEVER instantiate SBERT or GLiNER inside a loop.** All models are loaded once
   at the top of their step cell. Loading inside a loop adds 30+ seconds per slide.

3. **GLiNER replaced SIFRank entirely.** Step 3 uses
   `GLiNER.from_pretrained("urchade/gliner_large-v2.1")`. Do not reference SIFRankSqueezeBERT.

4. **The Groq API key is set ONCE in os.environ at the start of Step 4.**
   All subsequent steps read `os.environ["GROQ_API_KEY"]`. Never add another
   `getpass()` call. Step 9 uses the same key from the environment.

5. **Use MERGE, never CREATE in Neo4j.** Same concept appears across multiple slides.
   CREATE produces duplicates that break the graph.

6. **All concept names are lowercase and stripped** before validation, storage,
   and comparison. Enforce at Step 4 input and Step 7 write time.

7. **LLM temperature = 0 always.** Use `ChatGroq`. If call returns HTTP 429,
   automatically retry with `llama-3.1-8b-instant`. Free tier limit ≈ 30 req/min.
   Add `time.sleep(2)` between slides if rate limits are frequent.

8. **Never skip any of the 4 validation layers in Step 4.** All four must pass.
   Layer 4 (semantic post-validation) is especially critical — it catches direction
   errors and factual mistakes that the LLM prompt alone cannot prevent. The 5 rules
   (L4-2 through L4-5) are documented in detail in the Step 4 section above.

9. **Evidence sentences are exact quotes from the slide text.** The SBERT check in
   Layer 2 catches fabricated sentences. Never paraphrase evidence.

10. **Step 6 runs AFTER all keyphrases and triples are collected.** The Document
    Vocabulary requires the full document to be processed first.

11. **Do not use Wikipedia, DBpedia, or any external knowledge base** at any step.
    Closed-corpus constraint is a core research requirement for publication.

12. **The `needs_review` flag in Step 5** is informational only — flagged concepts
    are still kept in the graph. Only concepts below `WEIGHT_THRESHOLD` are pruned.

13. **OCR fallback lines are prefixed `> OCR:`** in Step 1 output. Step 2's
    `_classify_ocr_line()` handles re-classification. Do not strip this prefix before
    Step 2 runs.

14. **`source_type = "injected"` is a valid Keyphrase value.** Do not filter it out.
    It identifies keyphrases injected from the previous slide during Step 4's
    cross-slide enrichment. The `is_llm_slide` flag must exclude injected keyphrases
    when checking whether all keyphrases are LLM-sourced.

15. **Step 9 uses `llama-3.1-8b-instant`, not the large model.** Summaries are
    short narration tasks. The small model is sufficient and avoids burning rate limits.
    Do not switch to `llama-3.3-70b-versatile` for Step 9.

---

## 10. Known Limitations

- Diagram-only slides (image-only pages) produce no keyphrases or triples. Step 9
  emits a placeholder for these. This is expected behaviour, not a bug.
- Beamer LaTeX bullet markers (◮ ►) from some academic PDFs are not fully stripped
  by the current Step 2 regex, which degrades GLiNER keyphrase quality on those PDFs.
  Fix: extend `_BEAMER_BULLET_RE` to cover `◮` (U+25AE) explicitly.
- The cross-slide injection in Step 4 can backfire when the previous slide's top
  keyphrase is a constructor name or method signature (e.g. `arraylist(collection col)`).
  In that case the injected keyphrase causes reversed-direction triples. Fix: only inject
  keyphrases whose `source_type` is `heading` or `bullet` (not `table` or `body`).
- Step 7 uses single-item transactions (`session.execute_write`) — batching would be faster.
- No progress bar in Steps 5–6 for long PDFs.

---

## 11. Research Context

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
