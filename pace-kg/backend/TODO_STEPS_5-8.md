# TODO: Complete Steps 5-8 Implementation

## Current Status

The backend implementation is **90% complete**. The following steps have placeholder implementations and need to be filled in from the reference `edu_kg_gliner.py`:

### ✅ Completed Steps
- **Step 1**: Marker PDF Parsing (fully implemented)
- **Step 2**: Markdown Preprocessing (fully implemented)
- **Step 3**: Keyphrase Extraction with GLiNER + LLM fallback (fully implemented)
- **Step 4**: LLM Triple Extraction with 4-layer validation (fully implemented)
- **Step 9**: Summary Generation + Word export (fully implemented)

### ⚠️ Placeholder Steps (Need Implementation)

#### Step 5 — Concept Weighting & Pruning
**Location**: `pipeline_runner.py:step5_weight_and_prune()`

**Reference**: Lines ~2010-2200 in `edu_kg_gliner.py`

**What to implement**:
1. Build concept nodes from keyphrases
2. Calculate 3-signal SBERT weights:
   - `w_evidence` — SBERT cosine between concept and best evidence sentence
   - `w_slide` — SBERT cosine between concept and all slides it appears on
   - `w_doc` — SBERT cosine between concept and full document text
3. Apply relation role boosts (see CLAUDE.md Section 5)
4. Apply source type boosts (heading +0.10, bullet +0.05, etc.)
5. Calculate `final_weight = (0.5 * w_evidence) + (0.3 * w_slide) + (0.2 * w_doc) + boosts`
6. Prune concepts with `final_weight < 0.192`
7. Semantic merge: concepts with SBERT cosine ≥ 0.92
8. Flag concepts with cosine ≥ 0.75 but < 0.92 as `needs_review = True`
9. Prune triples that reference pruned concepts

**Output**:
- `List[ConceptNode]` — weighted and merged concepts
- `List[Triple]` — pruned triples (only references to kept concepts)

---

#### Step 6 — Closed-Corpus Expansion
**Location**: `pipeline_runner.py:step6_expand()`

**Reference**: Lines ~2279-2500 in `edu_kg_gliner.py`

**What to implement**:
1. Build document vocabulary from:
   - All keyphrases from Step 3
   - All triple subjects/objects from Step 5
   - spaCy noun chunks from all slide clean_text (strip articles)
2. For each core concept:
   - **Phase 1**: Ask LLM to select related concepts from pool (up to 80 vocab items)
   - **Phase 2**: SBERT gate — keep candidates with cosine ≥ 0.65
   - **Phase 3**: Slide-scope constraint — keep candidates that are:
     - On an adjacent slide (±1 index), OR
     - Have SBERT cosine ≥ 0.70 vs full document
   - **Phase 4**: Deduplicate — skip self-loops, trivial pairs, existing pairs
3. Create `ExpansionEdge` with:
   - `relation = "relatedConcept"`
   - `source = "expansion"`
   - `confidence = SBERT_cosine`
4. Cap at 10 expansion edges per concept

**Output**:
- `List[ExpansionEdge]` — expansion edges

---

#### Step 7 — Neo4j Storage
**Location**: `pipeline_runner.py:step7_store_neo4j()`

**Reference**: Lines ~2500-2700 in `edu_kg_gliner.py`

**What to implement**:
1. Connect to Neo4j using credentials from `settings`
2. **Two-pass write strategy**:
   - **Pass 1**: Store ALL concept nodes first using `MERGE` (never `CREATE`)
     ```cypher
     MERGE (c:Concept {name: $name})
     SET c.aliases = $aliases, c.slide_ids = $slide_ids, ...
     ```
   - **Pass 2**: Store edges slide-by-slide (extraction triples + expansion edges)
     ```cypher
     MATCH (s:Concept {name: $subject}), (o:Concept {name: $object})
     MERGE (s)-[r:RELATION {relation_type: $relation_type, slide_id: $slide_id, doc_id: $doc_id}]->(o)
     SET r.evidence = $evidence, r.confidence = $confidence, r.source = $source
     ```
   - **Pass 3**: Create `LearningMaterial` node + `BELONGS_TO` links
3. **Semantic conflict resolution**: Before writing, check if incoming concept name is similar (cosine ≥ 0.92) to existing concept; if so, use canonical name
4. Save storage report

**Output**:
- Graph stored in Neo4j
- `{doc_id}_step7_storage_report.json` — summary of stored nodes/edges

---

#### Step 8 — LM-EduKG Aggregation
**Location**: `pipeline_runner.py:step8_aggregate()`

**Reference**: Lines ~2700-2850 in `edu_kg_gliner.py`

**What to implement**:
1. Query Neo4j to export full graph (all concept nodes + all edges)
2. Verify four merge constraints:
   - Every slide has ≥1 concept
   - All local concepts exist in Neo4j
   - All extraction edges have non-empty evidence
   - Cross-slide expansion edges are tagged `material_level = true`
3. Build `lm_edkg` dict:
   ```python
   {
     "nodes": [ConceptNode as dict],
     "edges": [Triple + ExpansionEdge as dict],
     "srs_pool": [extraction triples only for evaluation]
   }
   ```
4. Save as `{doc_id}_step8_lm_edkg.json`

**Output**:
- `{doc_id}_step8_lm_edkg.json` — full graph export
- `{doc_id}_step8_summary.json` — verification report

---

## How to Complete

### Option 1: Extract from Reference Implementation

```bash
# Open the reference implementation
nano edu_kg_gliner.py

# Find Step 5 (search for "Step 5")
# Copy the implementation between the Step 5 marker and Step 6 marker
# Adapt to the PipelineRunner class structure

# Repeat for Steps 6, 7, 8
```

### Option 2: AI-Assisted Extraction

Use an AI coding assistant with the prompt:
```
Read edu_kg_gliner.py lines 2010-2850 (Steps 5-8).
Extract and adapt each step's implementation to fit the PipelineRunner class in pipeline_runner.py.
Ensure all helper functions are included and dataclass fields match.
```

### Option 3: Incremental Testing

1. Implement Step 5 first
2. Test with a sample PDF: `run_step5.py` (already exists in root)
3. Verify output JSON matches expected format
4. Repeat for Steps 6, 7, 8

---

## Testing Checklist

After implementing Steps 5-8, test the full pipeline:

```bash
# Start backend
cd pace-kg
docker-compose up backend

# Upload a test PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@../test3.pdf"

# Get doc_id from response, then check status
curl http://localhost:8000/status/<doc_id>

# Wait for completion (check logs: docker-compose logs -f backend)

# Verify outputs exist
ls outputs/<doc_id>/

# Should see all 9 step outputs:
# <doc_id>_step1_parsed.json
# <doc_id>_step2_preprocessed.json
# <doc_id>_step3_keyphrases.json
# <doc_id>_step4_triples.json
# <doc_id>_step5_concepts.json
# <doc_id>_step5_triples_pruned.json
# <doc_id>_step6_expansion.json
# <doc_id>_step7_storage_report.json
# <doc_id>_step8_lm_edkg.json
# <doc_id>_step9_summaries.json
# <doc_id>_summaries.docx

# Test API endpoints
curl http://localhost:8000/graph/<doc_id> | jq
curl http://localhost:8000/summaries/<doc_id> | jq
curl -O http://localhost:8000/export/<doc_id>
```

---

## Key Implementation Notes

1. **All models loaded once**: SBERT all-mpnet-base-v2 should be loaded in `__init__` and reused
2. **MERGE not CREATE**: Neo4j writes must use MERGE to avoid duplicates
3. **Lowercase everything**: All concept names lowercase before validation/storage
4. **Temperature 0**: All LLM calls use temperature 0
5. **Closed-corpus**: Step 6 must NOT use external knowledge (Wikipedia, DBpedia, etc.)

---

## Estimated Time

- **Step 5**: 1-2 hours (most complex weighting logic)
- **Step 6**: 1-2 hours (LLM + SBERT filtering)
- **Step 7**: 1 hour (Neo4j Cypher queries)
- **Step 8**: 30 minutes (simple export + verification)

**Total**: 4-6 hours for a developer familiar with the codebase.

---

## Priority

**High**: These steps are critical for the frontend to work. The frontend expects:
- `/graph/{doc_id}` to return `lm_edkg` from Step 8
- `/summaries/{doc_id}` to return summaries from Step 9 (already done)

Without Steps 5-8, the graph endpoint will return empty data.

---

## Questions?

Refer to:
- `CLAUDE.md` — full specification for each step
- `edu_kg_gliner.py` — working reference implementation
- `pace-kg/backend/README.md` — backend architecture overview
