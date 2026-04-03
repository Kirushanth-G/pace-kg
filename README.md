# PACE-KG

Automatically turns PDF lecture slides into a knowledge graph that students can explore.

Upload a PDF → get an interactive graph of concepts and their relationships → read slide-by-slide summaries.

## What it does

1. Extracts text from PDF slides (handles scanned pages too)
2. Finds key concepts using AI (GLiNER model)
3. Discovers relationships between concepts (e.g., "stack isPartOf data structures")
4. Builds a knowledge graph and stores it in Neo4j
5. Generates short summaries for each slide

The output is a web interface where you can:
- See all concepts as nodes in an interactive graph
- Click a concept to see which slide it came from and the evidence sentence
- Filter by slide, relationship type, or concept importance
- Read revision notes for each slide
- Download summaries as a Word document

## Architecture

```
┌─────────────────┐         ┌─────────────────────────────────────┐
│                 │         │  Vast.ai GPU Server (RTX 3090)      │
│  React Frontend │  HTTP   │  ┌─────────────────────────────┐    │
│  (runs locally) │ ──────► │  │  FastAPI Backend            │    │
│                 │         │  │  - Runs 9-step pipeline     │    │
└─────────────────┘         │  │  - Stores graph in Neo4j    │    │
                            │  │  - Returns JSON for frontend│    │
                            │  └─────────────────────────────┘    │
                            └─────────────────────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────────┐
                            │  External Services                  │
                            │  - Groq API (LLM calls)             │
                            │  - Neo4j AuraDB (graph storage)     │
                            └─────────────────────────────────────┘
```

**Why a GPU server?**  
The PDF parser (Marker) and concept extractor (GLiNER) need a GPU to run fast.

**Why Groq instead of local LLM?**  
The 70B model we use for relationship extraction won't fit on a 24GB GPU. Groq handles it.

## Pipeline Steps

| Step | What happens |
|------|--------------|
| 1 | Parse PDF with Marker, fallback to OCR for scanned pages |
| 2 | Clean up the text, remove noise, identify headings/bullets/tables |
| 3 | Extract key concepts using GLiNER (with LLM fallback) |
| 4 | Find relationships between concepts using LLM |
| 5 | Score concepts by importance, merge duplicates |
| 6 | Expand the graph with related concepts from the same document |
| 7 | Store everything in Neo4j |
| 8 | Export the final graph |
| 9 | Generate slide summaries and Word document |

## Project Structure

```
Edu-KG/
├── CLAUDE.md                 # Detailed docs for AI assistants
├── README.md                 # This file
├── pace-kg/
│   ├── backend/
│   │   ├── main.py           # FastAPI endpoints
│   │   ├── pipeline_runner.py # The 9-step pipeline
│   │   ├── api/models/       # Request/response schemas
│   │   ├── Dockerfile        # Container setup
│   │   └── requirements.txt  # Python dependencies
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx       # Main app with upload + results view
│   │   │   ├── components/
│   │   │   │   ├── GraphView.jsx    # D3.js graph visualization
│   │   │   │   └── SummaryPanel.jsx # Slide summaries
│   │   │   └── services/api.js      # Backend API calls
│   │   └── package.json
│   └── docker-compose.yml    # Run backend + Neo4j together
```

## Running It

### Backend (on GPU server)

```bash
cd pace-kg/backend

# Set up environment
cp .env.example .env
# Edit .env with your Groq API key and Neo4j credentials

# Run with Docker
docker build -t pace-kg-backend .
docker run --gpus all -p 8000:8000 --env-file .env pace-kg-backend
```

### Frontend (locally)

```bash
cd pace-kg/frontend

# Set backend URL
echo "VITE_API_URL=http://your-gpu-server:8000" > .env

npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/upload` | POST | Upload PDF, starts pipeline, returns job ID |
| `/status/{id}` | GET | Check pipeline progress |
| `/jobs` | GET | List completed jobs |
| `/graph/{id}` | GET | Get nodes and edges for visualization |
| `/summaries/{id}` | GET | Get slide summaries |
| `/export/{id}` | GET | Download Word document |

## Requirements

- GPU server with NVIDIA GPU (tested on RTX 3090)
- Groq API key (free at console.groq.com)
- Neo4j AuraDB instance (free tier works)
- Node.js 18+ for frontend

## Research Context

This is a research project comparing against the pipeline from:

> Ain et al. (2025) - "An Optimized Pipeline for Automatic Educational Knowledge Graph Construction"

Key differences from the original:
- Uses Marker instead of PDFMiner for better PDF parsing
- Uses GLiNER instead of SIFRank for concept extraction
- Uses LLM for relationship extraction instead of DBpedia Spotlight
- No external knowledge bases - everything comes from the PDF itself
