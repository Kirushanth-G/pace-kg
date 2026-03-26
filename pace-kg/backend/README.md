# PACE-KG Backend

Backend API for the PACE-KG (Pedagogically-Aware, Citation-Evidenced Knowledge Graph) system.

## Architecture

- **FastAPI** for REST API
- **GPU-accelerated** pipeline (Marker, GLiNER, SBERT)
- **Groq API** for LLM inference (Llama 3.3 70B and Llama 3.1 8B)
- **Neo4j AuraDB** for graph storage
- **Background tasks** for long-running pipeline execution

## API Endpoints

- `POST /upload` — Upload PDF and start pipeline
- `GET /status/{doc_id}` — Get pipeline progress
- `GET /graph/{doc_id}` — Get knowledge graph (nodes + edges)
- `GET /summaries/{doc_id}` — Get slide summaries
- `GET /export/{doc_id}` — Download Word document

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required:
- `GROQ_API_KEY` — Get from https://console.groq.com
- `NEO4J_URI` — Neo4j AuraDB connection string
- `NEO4J_USER` — Neo4j username (usually `neo4j`)
- `NEO4J_PASSWORD` — Neo4j password

### 2. Run with Docker (Recommended)

```bash
# From the pace-kg root directory
docker-compose up backend
```

The backend will be available at `http://localhost:8000`.

**Important**: Requires NVIDIA GPU with Docker GPU passthrough enabled.

### 3. Run Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set environment variables
export GROQ_API_KEY="your_key"
export NEO4J_URI="neo4j+s://xxxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Run
uvicorn main:app --reload
```

## Pipeline Steps

The backend runs a 9-step pipeline on each uploaded PDF:

1. **Marker PDF Parsing** — Deep-learning PDF parser with OCR fallback
2. **Markdown Preprocessing** — Structural parsing and noise removal
3. **Keyphrase Extraction** — GLiNER + LLM fallback
4. **Triple Extraction** — LLM with 4-layer validation
5. **Concept Weighting** — 3-signal SBERT weighting
6. **Closed-Corpus Expansion** — Internal concept expansion
7. **Neo4j Storage** — Store graph in Neo4j
8. **LM-EduKG Aggregation** — Export full graph
9. **Summary Generation** — Per-slide summaries + Word export

See `CLAUDE.md` in the root directory for full implementation details.

## Output Files

All outputs are saved in `./outputs/{doc_id}/`:

- `{doc_id}_step1_parsed.json` — Parsed slides
- `{doc_id}_step2_preprocessed.json` — Preprocessed content
- `{doc_id}_step3_keyphrases.json` — Extracted keyphrases
- `{doc_id}_step4_triples.json` — Extracted triples
- `{doc_id}_step5_concepts.json` — Weighted concepts
- `{doc_id}_step5_triples_pruned.json` — Pruned triples
- `{doc_id}_step6_expansion.json` — Expansion edges
- `{doc_id}_step7_storage_report.json` — Neo4j storage report
- `{doc_id}_step8_lm_edkg.json` — Full knowledge graph
- `{doc_id}_step9_summaries.json` — Slide summaries
- `{doc_id}_summaries.docx` — Word document

## GPU Requirements

- **Marker PDF parser** — CUDA-capable GPU, ~4GB VRAM
- **GLiNER large-v2.1** — CUDA-capable GPU, ~2GB VRAM
- **SBERT models** — CPU-capable, GPU accelerates

Tested on RTX 3090 (24GB).

## Troubleshooting

### Rate limits (Groq API)

The free tier has ~30 req/min. If you hit 429 errors:
- The code automatically falls back to the 8B model
- Add `time.sleep(2)` between slides if needed

### Out of memory

If GPU runs out of memory:
- Reduce batch sizes in GLiNER (not yet implemented)
- Use CPU for SBERT (slower but works)

### Neo4j connection errors

- Ensure Neo4j AuraDB instance is running
- Check firewall rules allow connections
- Verify credentials in `.env`

## License

See root directory.
