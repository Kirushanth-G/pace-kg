# CLAUDE.md — PACE-KG Project

> This file is the single source of truth for AI coding assistants (Claude, Copilot).
> Read this ENTIRE file before writing any code. Do not skip sections.

---

## 1. Project Overview

**PACE-KG** (Pedagogically-Aware, Citation-Evidenced Knowledge Graph) automatically
constructs Educational Knowledge Graphs (EduKGs) from PDF lecture slides.

It is an optimized redesign of the pipeline by Ain et al. (2025):
> arXiv:2509.05392 — "An Optimized Pipeline for Automatic Educational Knowledge Graph Construction"

PACE-KG keeps the proven architecture and surgically replaces the two weakest components.

### What is kept from the original
- SIFRankSqueezeBERT for keyphrase extraction (Step 3)
- SBERT all-mpnet-base-v2 for embeddings (Step 5)
- Redis + Celery worker queue architecture
- Neo4j as graph database
- Incremental slide-by-slide storage (available to learners per slide)

### What is replaced / new
- PDFMiner → pymupdf4llm (lightweight, 8GB-friendly parser)  [Step 1]
- NEW: Markdown Preprocessor            [Step 2]
- DBpedia Spotlight → LLM Triple Extraction [Step 4 — CORE NOVEL CONTRIBUTION]
- Wikipedia-dependent weighting → 3-Signal internal weighting [Step 5]
- Wikipedia dump + SPARQL → Closed-Corpus Expansion [Step 6 — NOVEL]

---

## 2. Repository Structure

```
pace-kg/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── main.py                          # FastAPI app entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upload.py                # POST /api/upload
│   │   │   ├── status.py                # GET  /api/status/{job_id}
│   │   │   ├── graph.py                 # GET  /api/graph/{doc_id}
│   │   │   ├── concept.py               # GET  /api/concept/{concept_id}
│   │   │   ├── learningpath.py          # GET  /api/learningpath/{doc_id}
│   │   │   ├── quiz.py                  # POST /api/quiz/{doc_id}
│   │   │   └── export.py                # GET  /api/export/{doc_id}
│   │   └── models/
│   │       ├── concept.py               # Pydantic: Concept, WeightedConcept
│   │       ├── triple.py                # Pydantic: Triple, ExpansionEdge
│   │       └── job.py                   # Pydantic: JobStatus
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py              # Runs all 8 steps in order
│   │   ├── step1_marker/
│   │   │   └── parser.py                # PDF → structured markdown per slide
│   │   ├── step2_preprocessor/
│   │   │   └── cleaner.py               # Markdown → clean typed text
│   │   ├── step3_keyphrase/
│   │   │   └── extractor.py             # SIFRankSqueezeBERT + adaptive filter
│   │   ├── step4_llm_extraction/
│   │   │   ├── extractor.py             # LLM triple extraction
│   │   │   ├── validator.py             # 3-layer hallucination prevention
│   │   │   └── prompts.py               # All LLM prompt templates
│   │   ├── step5_weighting/
│   │   │   └── weighter.py              # 3-signal SBERT weighting + pruning
│   │   ├── step6_expansion/
│   │   │   ├── expander.py              # Closed-corpus concept expansion
│   │   │   └── vocabulary.py            # Document vocabulary builder
│   │   ├── step7_storage/
│   │   │   ├── neo4j_client.py          # Neo4j connection + queries
│   │   │   ├── slide_kg.py              # Store Slide-EduKG per slide
│   │   │   └── conflict_resolver.py     # Semantic deduplication
│   │   └── step8_aggregation/
│   │       └── merger.py                # Merge Slide-EduKGs → LM-EduKG
│   │
│   ├── workers/
│   │   ├── celery_app.py                # Celery + Redis config
│   │   └── tasks.py                     # Celery task definitions
│   │
│   └── core/
│       ├── config.py                    # Settings via pydantic-settings
│       ├── embeddings.py                # SBERT singleton (load ONCE)
│       └── llm_client.py               # LLM client singleton
│
└── frontend/
    ├── package.json
    ├── Dockerfile
    └── src/
        ├── App.jsx
        ├── api/client.js                # All API calls
        ├── components/
        │   ├── Upload/
        │   │   ├── UploadView.jsx        # Drag-drop + progress
        │   │   └── ProgressTracker.jsx
        │   ├── Graph/
        │   │   ├── GraphView.jsx         # Cytoscape.js canvas
        │   │   ├── GraphControls.jsx     # Filter/search panel
        │   │   └── ConceptPanel.jsx      # Click-to-detail side panel
        │   ├── LearningPath/
        │   │   └── LearningPathView.jsx
        │   ├── Quiz/
        │   │   └── QuizView.jsx
        │   └── Dashboard/
        │       └── InstructorDashboard.jsx
        └── store/
            └── graphStore.js            # Zustand global state
```

---

## 3. Technology Stack

### Backend Python packages (requirements.txt)
```
fastapi
uvicorn[standard]
celery[redis]
redis
neo4j
pymupdf4llm
sentence-transformers
spacy
langchain
langchain-groq
mistune
pydantic-settings
python-multipart
```

After install: `python -m spacy download en_core_web_sm`

### Frontend packages (package.json)
```
react, react-dom
cytoscape
cytoscape-dagre
pdfjs-dist
zustand
axios
tailwindcss
```

---

## 4. Environment Variables (.env)

```bash
# LLM
GROQ_API_KEY=gsk_...        # get free key at console.groq.com
LLM_PRIMARY=llama-3.3-70b-versatile      # best quality, still free
LLM_FALLBACK=llama-3.1-8b-instant        # faster, use if rate limited

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Redis
REDIS_URL=redis://localhost:6379/0

# Pipeline thresholds
KEYPHRASE_MAX_CANDIDATES=30
KEYPHRASE_QUALITY_THRESHOLD=0.3
WEIGHT_PRUNING_THRESHOLD=0.192
EXPANSION_SIMILARITY_THRESHOLD=0.65
EXPANSION_MAX_RELATED=10
TRIPLE_CONFIDENCE_THRESHOLD=0.70
EVIDENCE_SIMILARITY_THRESHOLD=0.75
CONFLICT_MERGE_THRESHOLD=0.92

# Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=50
```

---

## 5. Data Models

Define these in `api/models/` and reuse across all pipeline steps.

```python
# api/models/triple.py
from dataclasses import dataclass, field
from typing import Literal, List

RelationType = Literal[
    "isPrerequisiteOf",
    "isDefinedAs",
    "isExampleOf",
    "contrastedWith",
    "appliedIn",
    "isPartOf",
    "causeOf",
    "isGeneralizationOf",
]

@dataclass
class Triple:
    subject: str        # lowercase, stripped
    relation: str       # one of RelationType
    object: str         # lowercase, stripped
    evidence: str       # exact sentence from PDF — NEVER paraphrased
    confidence: float   # 0.0 to 1.0
    slide_id: str
    doc_id: str
    source: str = "extraction"

@dataclass
class ExpansionEdge:
    subject: str
    object: str
    relation: str = "relatedConcept"
    source: str = "expansion"
    confidence: float = 0.0
    slide_id: str = ""
    doc_id: str = ""

# api/models/concept.py
@dataclass
class Keyphrase:
    phrase: str          # lowercase, stripped
    score: float
    source_type: str     # heading | body | bullet | table | caption
    slide_id: str
    doc_id: str
    appears_in: str      # sentence containing this phrase

@dataclass
class WeightedConcept:
    name: str
    final_weight: float
    slide_id: str
    doc_id: str
    source_type: str
    keyphrase_score: float
    triples: List[Triple] = field(default_factory=list)

# api/models/job.py
@dataclass
class JobStatus:
    job_id: str
    doc_id: str
    status: str          # queued | parsing | preprocessing | processing | expanding | aggregating | complete | failed
    slides_total: int = 0
    slides_completed: int = 0
    current_step: str = ""
    error: str = ""
```

---

## 6. Pipeline Implementation — Step by Step

---

### STEP 1 — pymupdf4llm Parsing
**File:** `pipeline/step1_marker/parser.py`

**What it does:** PDF file → list of SlideMarkdown objects (one per page)

**Key points:**
- Use `pymupdf4llm.to_markdown(file_path, page_chunks=True)` — returns a list of dicts, one per page
- Each chunk has `"text"` (markdown string) and `"metadata"` with `"page"` (0-indexed)
- Cache output: compute `sha256(pdf_bytes)`, cache result as JSON keyed by hash
- If cache hit: return cached result immediately, skip re-parsing
- Assign slide_id as zero-padded: `slide_001`, `slide_002` etc.
- **Memory footprint: ~50 MB — no ML models required. Runs on 8GB machines.**
- Text embedded inside images is NOT extracted. For image-heavy PDFs, use the Colab+Marker workflow described in `pace-kg/COLAB_MARKER_GUIDE.md` to pre-generate a cached JSON, then call `load_parsed_json()` instead of `parse_pdf()`.

**Output per slide:**
```python
@dataclass
class SlideMarkdown:
    slide_id: str
    page_number: int
    raw_markdown: str
    doc_id: str
```

**Do NOT clean text here** — that is Step 2.

---

### STEP 2 — Markdown Preprocessor
**File:** `pipeline/step2_preprocessor/cleaner.py`

**What it does:** Raw markdown per slide → clean typed content tree

**Stage 1 — Structural parsing with mistune:**
Parse markdown tags into typed buckets:
- Lines starting with `#` or `##` → `headings` (strip `#` symbols)
- Lines starting with `-` or `*` → `bullets` (strip bullet char)
- Table rows `| cell | cell |` → `table_cells` (split on `|`, strip whitespace)
- Lines starting with `>` → `captions` (strip `>`)
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
Discard any text block fully matching a pattern (case-insensitive).

**Stage 3 — Cross-slide repetition filter (run AFTER all slides parsed):**
This is a post-processing pass. Collect all text blocks from all slides.
If any block appears in > 50% of slides → mark as noise, remove from all slides.

**Stage 4 — Assemble clean_text:**
```python
clean_text = " ".join(headings + body_text + bullets + table_cells + captions)
```

**Important:** Store `heading_phrases` (just the heading strings) separately.
These are used for the +0.20 score boost in Step 3.

---

### STEP 3 — Keyphrase Extraction
**File:** `pipeline/step3_keyphrase/extractor.py`

**What it does:** SlideContent → List[Keyphrase] using SIFRankSqueezeBERT

**Reference:** Use the SIFRank logic from `coursemapper-kg/app/algorithms/sifrank/`.
The model used is `sentence-transformers/all-MiniLM-L6-v2` in their implementation.

**Adaptive filter pipeline (apply IN THIS ORDER):**

1. Extract up to `KEYPHRASE_MAX_CANDIDATES=30` from SIFRankSqueezeBERT
2. Drop phrases with score < `KEYPHRASE_QUALITY_THRESHOLD=0.3`
3. spaCy linguistic filter:
   ```python
   def is_valid(phrase):
       doc = nlp(phrase)
       has_noun = any(t.pos_ in ["NOUN", "PROPN"] for t in doc)
       all_stop = all(t.is_stop for t in doc)
       return has_noun and not all_stop and len(phrase.strip()) >= 3
   ```
4. Noun-chunk cross-validation:
   ```python
   def in_noun_chunks(phrase, clean_text):
       doc = nlp(clean_text)
       chunks = [c.text.lower() for c in doc.noun_chunks]
       return phrase.lower() in chunks
   ```
5. Assign source_type by checking which bucket the phrase appears in
   (check headings first, then bullets, tables, captions, body — in that order)
6. Apply heading boost: if source_type == "heading", score = min(score + 0.20, 1.0)

**The appears_in field:**
Find the sentence in clean_text that contains the phrase. Use spaCy sentence
segmentation. If not found, use the full clean_text as fallback.

---

### STEP 4 — LLM Triple Extraction
**File:** `pipeline/step4_llm_extraction/`

**What it does:** Keyphrases + SlideContent → List[Triple]

This is the CORE NOVEL CONTRIBUTION. Read carefully.

#### prompts.py — copy exactly

```python
SYSTEM_PROMPT = """You are a knowledge graph construction assistant for educational materials.

STRICT RULES — violating any rule invalidates your entire response:
1. ONLY use concepts from the KEYPHRASE LIST as subject and object. Nothing else.
2. Every triple MUST include an EXACT sentence copied from the SLIDE TEXT as evidence.
3. If no supporting sentence exists in the slide text, omit that triple entirely.
4. Use ONLY these relation types:
   isPrerequisiteOf   - A must be understood before B
   isDefinedAs        - formal definition of A is given
   isExampleOf        - A is a specific example of B
   contrastedWith     - A and B are explicitly compared
   appliedIn          - A is used/applied in context B
   isPartOf           - A is a structural component of B
   causeOf            - A causes or leads to B
   isGeneralizationOf - A is a broader category including B
5. Return ONLY a valid JSON array. No markdown, no explanation, no preamble.
6. If no valid triples found, return: []"""

USER_PROMPT = """KEYPHRASE LIST (subject and object MUST come from this list):
{keyphrases}

SLIDE TEXT:
{slide_text}

Extract all valid triples as JSON array. Each item:
{{
  "subject": "exact phrase from keyphrase list",
  "relation": "one of the 8 relation types",
  "object": "exact phrase from keyphrase list (different from subject)",
  "evidence": "exact sentence copied from slide text above",
  "confidence": 0.0 to 1.0
}}"""
```

#### extractor.py

```python
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import json

class TripleExtractor:
    def __init__(self, config):
        self.llm = ChatGroq(
            model=config.LLM_PRIMARY,
            temperature=0,
            # Groq supports JSON mode on Llama 3.x models
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def extract(self, keyphrases, slide_content):
        kp_list = "\n".join(f"- {k.phrase}" for k in keyphrases)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(
                keyphrases=kp_list,
                slide_text=slide_content.clean_text
            ))
        ]
        try:
            response = self.llm.invoke(messages)
            raw = json.loads(response.content)
            if isinstance(raw, list):
                return raw
            for v in raw.values():
                if isinstance(v, list):
                    return v
        except Exception:
            return []
        return []
```

#### validator.py — THREE LAYERS, all required

```python
class TripleValidator:
    VALID_RELATIONS = {
        "isPrerequisiteOf", "isDefinedAs", "isExampleOf",
        "contrastedWith", "appliedIn", "isPartOf",
        "causeOf", "isGeneralizationOf"
    }

    def validate(self, raw, keyphrases, slide_text, slide_id, doc_id):
        subject = str(raw.get("subject", "")).lower().strip()
        obj     = str(raw.get("object",  "")).lower().strip()
        relation= str(raw.get("relation","")).strip()
        evidence= str(raw.get("evidence","")).strip()
        conf    = float(raw.get("confidence", 0))

        kp_lower = {k.phrase.lower() for k in keyphrases}

        # Layer 1: Anchor constraint
        if subject not in kp_lower: return None
        if obj     not in kp_lower: return None
        if subject == obj:           return None
        if relation not in self.VALID_RELATIONS: return None

        # Layer 2: Evidence verification
        if not evidence: return None
        ev_emb   = sbert.encode(evidence,   convert_to_tensor=True)
        text_emb = sbert.encode(slide_text, convert_to_tensor=True)
        if float(util.cos_sim(ev_emb, text_emb)) < config.EVIDENCE_SIMILARITY_THRESHOLD:
            return None

        # Layer 3: Confidence threshold
        if conf < config.TRIPLE_CONFIDENCE_THRESHOLD: return None

        return Triple(subject=subject, relation=relation, object=obj,
                      evidence=evidence, confidence=conf,
                      slide_id=slide_id, doc_id=doc_id)
```

---

### STEP 5 — Concept Weighting & Pruning
**File:** `pipeline/step5_weighting/weighter.py`

**What it does:** Triples + slide/doc text → WeightedConcept list with pruned low-weight items

**Exact weight formula:**
```
final_weight = (0.5 x w_evidence) + (0.3 x w_slide) + (0.2 x w_doc)
               + relation_role_boost + source_type_boost

w_evidence = cosine_sim(SBERT(concept_name), SBERT(evidence_sentence))
w_slide    = cosine_sim(SBERT(concept_name), SBERT(slide_text))
w_doc      = cosine_sim(SBERT(concept_name), SBERT(full_doc_text))
```

**Relation role boosts:**
```python
RELATION_BOOST = {
    ("object",  "isDefinedAs"):        +0.15,
    ("object",  "isGeneralizationOf"): +0.10,
    ("subject", "isPrerequisiteOf"):   +0.10,
    ("subject", "causeOf"):            +0.05,
    ("subject", "contrastedWith"):     +0.05,
    ("object",  "contrastedWith"):     +0.05,
    ("subject", "isExampleOf"):        -0.05,
}
# Sum all applicable boosts, cap at +0.20
```

**Source type boosts:**
```python
SOURCE_BOOST = {
    "heading": +0.10,
    "bullet":  +0.05,
    "table":   +0.05,
    "body":    +0.00,
    "caption": -0.05,
}
```

**Pruning:**
- Remove concepts with final_weight < WEIGHT_PRUNING_THRESHOLD (0.192)
- When removing a concept, also remove ALL triples referencing it
- A triple is removed if its subject OR object is pruned

---

### STEP 6 — Closed-Corpus Concept Expansion
**File:** `pipeline/step6_expansion/`

**CRITICAL:** This step runs AFTER all slides are processed, not per-slide.
It needs the complete Document Vocabulary which only exists after full processing.

**What it does:** For each surviving concept, find related concepts FROM THE DOCUMENT ONLY.

#### vocabulary.py — Build Document Vocabulary from 3 sources

```python
def build_vocabulary(all_concepts, all_triples, all_slide_contents, nlp):
    vocab = set()
    # Source 1: all surviving concept names
    vocab.update(c.name for c in all_concepts)
    # Source 2: all triple subjects and objects
    for t in all_triples:
        vocab.add(t.subject)
        vocab.add(t.object)
    # Source 3: noun chunks from full document text
    full_text = " ".join(sc.clean_text for sc in all_slide_contents)
    doc = nlp(full_text)
    for chunk in doc.noun_chunks:
        c = chunk.text.lower().strip()
        if len(c) >= 3:
            vocab.add(c)
    return vocab
```

#### expander.py — Four phases

```python
EXPANSION_PROMPT = """Main concept: '{concept}'
Slide context: '{context}'

Candidate pool — SELECT FROM THIS LIST ONLY:
{candidates}

Which concepts from this list are most educationally related to '{concept}'?
DO NOT suggest anything not in the list.
Return max {max_n} as a JSON array of strings. Example: ["concept a", "concept b"]"""

def expand_concept(mc, vocab, all_concepts, slide_content, slide_contents, doc_text, sbert, llm, config):

    # Phase 1: candidate pool = vocabulary minus the concept itself
    pool = vocab - {mc.name}

    # Phase 2: LLM selects from closed pool
    prompt = EXPANSION_PROMPT.format(
        concept=mc.name,
        context=slide_content.clean_text[:400],
        candidates="\n".join(f"- {c}" for c in sorted(pool)),
        max_n=config.EXPANSION_MAX_RELATED
    )
    try:
        response = llm.invoke(prompt)
        selected = json.loads(response.content)
        if not isinstance(selected, list):
            return []
    except:
        return []

    # Phase 3: SBERT similarity gate
    mc_emb = sbert.encode(mc.name, convert_to_tensor=True)
    passed = []
    for candidate in selected:
        if candidate not in pool:
            continue  # reject anything not in pool (safety)
        cand_emb = sbert.encode(candidate, convert_to_tensor=True)
        sim = float(util.cos_sim(mc_emb, cand_emb))
        if sim >= config.EXPANSION_SIMILARITY_THRESHOLD:
            passed.append((candidate, sim))

    # Phase 4: Slide scope constraint
    curr_idx = next((i for i,sc in enumerate(slide_contents)
                     if sc.slide_id == slide_content.slide_id), 0)
    adjacent_concepts = {
        c.name for c in all_concepts
        if any(abs(i - curr_idx) <= 1 and sc.slide_id == c.slide_id
               for i, sc in enumerate(slide_contents))
    }
    high_weight_concepts = {c.name for c in all_concepts if c.final_weight > 0.7}
    in_scope = adjacent_concepts | high_weight_concepts

    return [
        ExpansionEdge(subject=mc.name, object=cand, confidence=sim,
                      slide_id=slide_content.slide_id, doc_id=slide_content.doc_id)
        for cand, sim in passed if cand in in_scope
    ]
```

---

### STEP 7 — Slide-EduKG Storage in Neo4j
**File:** `pipeline/step7_storage/`

**Store immediately after each slide is processed.** Do not wait for full document.

#### neo4j_client.py — Run these on startup

```cypher
CREATE CONSTRAINT concept_unique IF NOT EXISTS
  FOR (c:Concept) REQUIRE (c.name, c.doc_id) IS UNIQUE;

CREATE INDEX concept_doc IF NOT EXISTS
  FOR (c:Concept) ON (c.doc_id);

CREATE INDEX concept_weight IF NOT EXISTS
  FOR (c:Concept) ON (c.final_weight);
```

#### slide_kg.py — Key Cypher queries

```python
# MERGE node — never CREATE — same concept appears in multiple slides
MERGE_CONCEPT = """
MERGE (c:Concept {name: $name, doc_id: $doc_id})
ON CREATE SET
    c.aliases = [],
    c.slide_ids = [$slide_id],
    c.source_type = $source_type,
    c.keyphrase_score = $keyphrase_score,
    c.final_weight = $final_weight
ON MATCH SET
    c.slide_ids = CASE
        WHEN $slide_id IN c.slide_ids THEN c.slide_ids
        ELSE c.slide_ids + [$slide_id] END,
    c.final_weight = CASE
        WHEN $final_weight > c.final_weight THEN $final_weight
        ELSE c.final_weight END
"""

CREATE_EDGE = """
MATCH (a:Concept {name: $subject, doc_id: $doc_id})
MATCH (b:Concept {name: $object,  doc_id: $doc_id})
MERGE (a)-[r:RELATION {relation_type: $relation_type, slide_id: $slide_id}]->(b)
SET r.evidence = $evidence,
    r.confidence = $confidence,
    r.source = $source
"""
```

**CRITICAL: Use batch writes.** Collect all nodes and edges for one slide,
write in a single transaction. Never write one node/edge per transaction.

#### conflict_resolver.py

```python
def find_merge_target(new_name, existing_names, sbert, threshold=0.92):
    """Returns name to merge into, or None if concept is genuinely new."""
    if not existing_names:
        return None
    new_emb = sbert.encode(new_name, convert_to_tensor=True)
    ex_embs = sbert.encode(existing_names, convert_to_tensor=True)
    sims = util.cos_sim(new_emb, ex_embs)[0]
    idx = int(sims.argmax())
    if float(sims[idx]) >= threshold:
        return existing_names[idx]
    return None
```

---

### STEP 8 — LM-EduKG Aggregation
**File:** `pipeline/step8_aggregation/merger.py`

**What it does:** Creates the LearningMaterial node and enforces 4 merge constraints.

Steps 1-7 already used MERGE queries, so most aggregation is done.
This step finalises the graph and adds material-level metadata.

```python
CREATE_MATERIAL_NODE = """
MERGE (m:LearningMaterial {doc_id: $doc_id})
SET m.title = $title,
    m.total_slides = $total_slides,
    m.created_at = datetime()
"""

LINK_ALL_CONCEPTS = """
MATCH (c:Concept {doc_id: $doc_id})
MATCH (m:LearningMaterial {doc_id: $doc_id})
MERGE (c)-[:BELONGS_TO]->(m)
"""

# After linking, compute and store material-level stats
COMPUTE_STATS = """
MATCH (m:LearningMaterial {doc_id: $doc_id})
MATCH (c:Concept {doc_id: $doc_id})
MATCH ()-[r:RELATION]->()
  WHERE r.slide_id STARTS WITH $doc_id
SET m.total_concepts = count(DISTINCT c),
    m.total_relations = count(DISTINCT r)
"""
```

**Four merge constraints (verify all pass):**
1. Every slide has at least one concept stored ← check Neo4j
2. All slide-level concepts exist at material level ← guaranteed by MERGE
3. All evidence fields are non-null ← check before Step 7 write
4. Cross-slide expansion edges exist ← check ExpansionEdge count > 0

---

## 7. API Routes

### POST /api/upload
```python
@router.post("/api/upload")
async def upload(file: UploadFile):
    # Validate: PDF only, size < MAX_UPLOAD_SIZE_MB
    # Save: UPLOAD_DIR / {uuid}.pdf
    # Enqueue: process_pdf.delay(job_id, file_path)
    # Return: {job_id, status: "queued"}
```

### GET /api/status/{job_id}
```python
# Return JobStatus from Redis
# Frontend polls this every 2 seconds
# Must update after EVERY slide, not just at end
```

### GET /api/graph/{doc_id}
```python
# Return Cytoscape-compatible JSON:
{
  "nodes": [{"data": {"id": "...", "label": "...", "weight": 0.9, "source_type": "heading"}}],
  "edges": [{"data": {"id": "...", "source": "...", "target": "...",
                      "relation": "isPrerequisiteOf", "evidence": "...", "confidence": 0.92}}]
}
```

### GET /api/learningpath/{doc_id}
```python
# Query all isPrerequisiteOf edges
# Topological sort using networkx
# Return ordered list of concept names with prerequisite info
```

### POST /api/quiz/{doc_id}
```python
QUIZ_TEMPLATES = {
    "isDefinedAs":        "What is {object}?",
    "isPrerequisiteOf":   "What must you understand before learning {object}?",
    "isExampleOf":        "{subject} is an example of what broader concept?",
    "contrastedWith":     "What is the key difference between {subject} and {object}?",
    "appliedIn":          "In what context is {subject} typically applied?",
    "causeOf":            "What does {subject} lead to?",
    "isGeneralizationOf": "{subject} is a generalization of what concept?",
    "isPartOf":           "What is {subject} a component of?",
}
# Fill templates from Neo4j triples
# Return [{question, relation_type, evidence, answer_hint}]
```

---

## 8. Celery Worker

### workers/celery_app.py
```python
from celery import Celery
celery = Celery("pace_kg", broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_serializer = "json"
celery.conf.task_track_started = True
celery.conf.task_acks_late = True        # re-queue on worker crash
celery.conf.worker_prefetch_multiplier = 1
```

### pipeline/orchestrator.py — CRITICAL ORDERING

```python
class PipelineOrchestrator:
    def run(self, file_path, doc_id, job_id):

        # Steps 1-2: parse and preprocess entire PDF first
        self.update_status(job_id, "parsing")
        slides_md = step1_parse(file_path, doc_id)

        self.update_status(job_id, "preprocessing")
        slide_contents = step2_preprocess(slides_md)
        # Step 2 includes cross-slide repetition filter across ALL slides

        doc_text = " ".join(sc.clean_text for sc in slide_contents)
        all_weighted = []
        all_triples = []

        # Steps 3-7: process slide by slide
        for slide in slide_contents:
            keyphrases = step3_extract(slide)
            raw_triples = step4_extract_triples(keyphrases, slide)
            weighted = step5_weight_and_prune(raw_triples, slide, doc_text)
            all_weighted.extend(weighted)
            all_triples.extend(t for c in weighted for t in c.triples)
            step7_store_slide(weighted, slide)          # store immediately
            self.update_slide_progress(job_id, slide.slide_id)

        # Step 6: AFTER all slides — needs full document vocabulary
        self.update_status(job_id, "expanding")
        expansion_edges = step6_expand(all_weighted, all_triples, slide_contents, doc_text)
        step7_store_expansion_edges(expansion_edges, doc_id)

        # Step 8: aggregate
        self.update_status(job_id, "aggregating")
        step8_aggregate(doc_id, len(slide_contents))

        self.update_status(job_id, "complete")
```

---

## 9. Frontend Key Patterns

### Polling for real-time slide progress
```javascript
// UploadView.jsx
useEffect(() => {
  if (!jobId) return;
  const interval = setInterval(async () => {
    const { data } = await getStatus(jobId);
    setProgress(data);
    if (data.status === "complete") {
      clearInterval(interval);
      navigate(`/graph/${data.doc_id}`);
    }
    if (data.status === "failed") {
      clearInterval(interval);
      setError(data.error);
    }
  }, 2000);
  return () => clearInterval(interval);
}, [jobId]);
```

### Cytoscape.js edge colors by relation type
```javascript
const RELATION_COLORS = {
  isPrerequisiteOf:   "#E74C3C",   // red
  isDefinedAs:        "#3498DB",   // blue
  isExampleOf:        "#2ECC71",   // green
  contrastedWith:     "#F39C12",   // orange
  appliedIn:          "#9B59B6",   // purple
  isPartOf:           "#1ABC9C",   // teal
  causeOf:            "#E67E22",   // dark orange
  isGeneralizationOf: "#34495E",   // dark grey
  relatedConcept:     "#BDC3C7",   // light grey (expansion edges)
};

// Node size by weight
"width":  "mapData(weight, 0, 1, 20, 70)",
"height": "mapData(weight, 0, 1, 20, 70)",
```

### Concept Detail Panel — what to show on node click
```javascript
// ConceptPanel.jsx
// Fetch: GET /api/concept/{conceptId}?doc_id={docId}
// Display:
// - Concept name + aliases
// - Weight + source_type badge
// - All outgoing edges with relation type + evidence sentence
// - All incoming edges with relation type + evidence sentence
// - "Jump to Slide" button per evidence → scroll PDF.js to page_number
// - "Appears in slides" list
```

---

## 10. Docker Compose

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [redis, neo4j]
    volumes: ["./uploads:/app/uploads"]
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build: ./backend
    command: celery -A workers.celery_app worker --loglevel=info --concurrency=2
    env_file: .env
    depends_on: [redis, neo4j]
    volumes: ["./uploads:/app/uploads"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      VITE_API_URL: http://localhost:8000

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/your_password
    volumes: ["neo4j_data:/data"]

volumes:
  neo4j_data:
```

---

## 11. Core Singletons — Load Once, Reuse Everywhere

### core/embeddings.py
```python
from sentence_transformers import SentenceTransformer
_model = None

def get_sbert():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model
```

### core/llm_client.py
```python
from langchain_groq import ChatGroq

_primary_llm = None
_fallback_llm = None

def get_llm(config):
    global _primary_llm
    if _primary_llm is None:
        _primary_llm = ChatGroq(
            model=config.LLM_PRIMARY,      # llama-3.3-70b-versatile
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    return _primary_llm

def get_fallback_llm(config):
    """Use this if Groq rate limits you (HTTP 429)."""
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = ChatGroq(
            model=config.LLM_FALLBACK,     # llama-3.1-8b-instant
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    return _fallback_llm
```

---

## 12. Rules for AI Assistants — Read Before Writing Any Code

1. **NEVER instantiate SBERT inside a loop or per-request.** Use `get_sbert()` singleton.
   Loading SBERT per slide adds 30+ seconds per slide.

2. **Step 6 runs AFTER all slides, not per-slide.** The Document Vocabulary only exists
   after the full PDF is processed. See orchestrator.py ordering.

3. **Use MERGE, never CREATE in Neo4j.** Same concept appears across multiple slides.
   CREATE produces duplicates that break the graph.

4. **Batch Neo4j writes.** One transaction per slide with all nodes + edges.
   One write per node is too slow for production.

5. **The 15-keyphrase cap is removed.** Use adaptive filter up to 30.
   Never hardcode 15 anywhere.

6. **All concept names are lowercase and stripped** before validation, storage, and
   comparison. Enforce this at Step 4 input and Step 7 write time.

7. **LLM temperature = 0 always.** Use ChatGroq, never ChatOpenAI or ChatOllama.
   If a call returns HTTP 429 (rate limit), automatically retry with get_fallback_llm().
   Free tier limit is 30 requests/minute — the pipeline fits within this comfortably.

8. **Never skip any of the 3 validation layers in Step 4.** All three must pass.

9. **The expansion LLM prompt must include the full candidate pool.** If the pool is
   very large (>200 items), chunk into batches of 100 and union the results.

10. **Evidence sentences are exact quotes.** Never paraphrase in the validator.
    The SBERT check catches fabricated sentences.

11. **Update job status after EVERY slide**, not just at start and end.
    Frontend polls every 2 seconds — users expect slide-level progress.

12. **Do not use Wikipedia, DBpedia, or any external knowledge base** at any step.
    The closed-corpus constraint is a research requirement, not an optimization.

---

## 13. Development Order

```
Week 1:  docker-compose up (Neo4j + Redis running)
         core/config.py, core/embeddings.py, core/llm_client.py
         Step 1: pymupdf4llm parser + caching (runs on 8GB, <2s per PDF)
         Step 2: Markdown preprocessor
         Test: parse a real 10-slide PDF end-to-end
         Optional: for image-heavy PDFs follow COLAB_MARKER_GUIDE.md

Week 2:  Step 3: Keyphrase extractor with adaptive filter
         Step 4: LLM extractor + all 3 validator layers
         Test: extract triples from 3 slides, inspect JSON output manually

Week 3:  Step 5: Concept weighting + pruning
         Step 6: Vocabulary builder + 4-phase expander
         Test: check that no out-of-vocabulary concept appears in expansion output

Week 4:  Step 7: Neo4j storage + conflict resolver
         Step 8: Aggregation
         Celery worker + orchestrator
         Test: full pipeline on one PDF, inspect Neo4j browser

Week 5:  FastAPI routes (upload, status, graph, concept)
         Frontend: Upload view + status polling
         Frontend: Graph view with Cytoscape.js

Week 6:  Frontend: Concept panel + Learning path + Quiz
         Instructor dashboard
         End-to-end test with evaluation PDF from original paper
         Performance benchmark: time per slide vs 2.3s baseline
```

---

## 14. Research Context

This project is for academic publication comparing against:
- **Baseline:** Ain et al. (2025) optimized pipeline — accuracy 0.47
- **Target:** PACE-KG accuracy > 0.47 using same SRS evaluation method

Two research claims must be preserved in every implementation decision:
1. **Evidence anchoring** — every triple must have a source sentence from the PDF
2. **Closed-corpus expansion** — no concept from outside the PDF enters the KG

The evaluation PDF, expert annotators, and SRS sampling method must match
the original paper exactly. Keep `evaluation/` code completely separate from
`pipeline/` code.
