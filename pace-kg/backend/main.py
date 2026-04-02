"""FastAPI application entry point."""

import json
import uuid
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.models.schemas import (
    ConceptNode,
    Edge,
    GraphResponse,
    JobStatus,
    SlideSummary,
    SummariesResponse,
    UploadResponse,
)
from core.config import settings
from pipeline_runner import PipelineRunner

app = FastAPI(
    title="PACE-KG",
    description="Pedagogically-Aware, Citation-Evidenced Knowledge Graph API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracker (replace with Redis/DB in production)
jobs: Dict[str, JobStatus] = {}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF and start the pipeline processing.

    Returns a doc_id that can be used to track progress and retrieve results.
    """
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Generate doc_id
    doc_id = str(uuid.uuid4())

    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / f"{doc_id}.pdf"

    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Check file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.max_upload_size_mb:
        pdf_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size_mb:.1f}MB) exceeds limit ({settings.max_upload_size_mb}MB)",
        )

    # Initialize job status
    jobs[doc_id] = JobStatus(
        doc_id=doc_id,
        status="pending",
        current_step="Initializing",
        progress=0.0,
        error=None,
    )

    # Start background task
    background_tasks.add_task(run_pipeline, doc_id, str(pdf_path))

    return UploadResponse(
        doc_id=doc_id, status="pending", message=f"Pipeline started for {file.filename}"
    )


def run_pipeline(doc_id: str, pdf_path: str):
    """
    Background task to run the full 9-step pipeline.
    Updates job status as it progresses.
    """
    try:
        # Update status
        jobs[doc_id].status = "running"
        jobs[doc_id].current_step = "Step 1: Parsing PDF"
        jobs[doc_id].progress = 0.1

        # Create runner
        output_dir = Path(settings.output_dir) / doc_id
        runner = PipelineRunner(doc_id, pdf_path, str(output_dir))

        # Step 1
        slides_md = runner.step1_parse_pdf()
        jobs[doc_id].current_step = "Step 2: Preprocessing"
        jobs[doc_id].progress = 0.2

        # Step 2
        slides_content = runner.step2_preprocess(slides_md)
        jobs[doc_id].current_step = "Step 3: Extracting Keyphrases"
        jobs[doc_id].progress = 0.3

        # Step 3
        keyphrases = runner.step3_extract_keyphrases(slides_content)
        jobs[doc_id].current_step = "Step 4: Extracting Triples"
        jobs[doc_id].progress = 0.4

        # Step 4
        triples = runner.step4_extract_triples(slides_content, keyphrases)
        jobs[doc_id].current_step = "Step 5: Weighting Concepts"
        jobs[doc_id].progress = 0.5

        # Step 5
        concepts, pruned_triples = runner.step5_weight_and_prune(
            triples, keyphrases, slides_content
        )
        jobs[doc_id].current_step = "Step 6: Expanding Graph"
        jobs[doc_id].progress = 0.6

        # Step 6
        expansions = runner.step6_expand(
            concepts, pruned_triples, slides_content, keyphrases
        )
        jobs[doc_id].current_step = "Step 7: Storing in Neo4j"
        jobs[doc_id].progress = 0.7

        # Step 7
        runner.step7_store_neo4j(concepts, pruned_triples, expansions, slides_content)
        jobs[doc_id].current_step = "Step 8: Aggregating Graph"
        jobs[doc_id].progress = 0.8

        # Step 8
        lm_edkg = runner.step8_aggregate()
        jobs[doc_id].current_step = "Step 9: Generating Summaries"
        jobs[doc_id].progress = 0.9

        # Step 9
        summaries = runner.step9_generate_summaries(slides_content, lm_edkg)

        # Complete
        jobs[doc_id].status = "completed"
        jobs[doc_id].current_step = "Done"
        jobs[doc_id].progress = 1.0

    except Exception as e:
        jobs[doc_id].status = "failed"
        jobs[doc_id].error = str(e)
        print(f"Pipeline failed for {doc_id}: {e}")
        import traceback

        traceback.print_exc()


@app.get("/status/{doc_id}", response_model=JobStatus)
async def get_status(doc_id: str):
    """
    Get the current status of a pipeline job.

    Returns progress information and current step.
    """
    if doc_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[doc_id]


@app.get("/graph/{doc_id}", response_model=GraphResponse)
async def get_graph(doc_id: str):
    """
    Get the knowledge graph (nodes and edges) for rendering.

    Returns the full LM-EduKG from Step 8.
    """
    if doc_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[doc_id].status != "completed":
        raise HTTPException(status_code=400, detail="Pipeline not yet completed")

    # Load step8 output
    output_dir = Path(settings.output_dir) / doc_id
    lm_edkg_path = output_dir / f"{doc_id}_step8_lm_edkg.json"

    if not lm_edkg_path.exists():
        raise HTTPException(status_code=404, detail="Graph data not found")

    with open(lm_edkg_path, "r", encoding="utf-8") as f:
        lm_edkg = json.load(f)

    # Parse nodes and edges
    nodes = [ConceptNode(**n) for n in lm_edkg.get("nodes", [])]
    edges = [Edge(**e) for e in lm_edkg.get("edges", [])]

    return GraphResponse(doc_id=doc_id, nodes=nodes, edges=edges)


@app.get("/summaries/{doc_id}", response_model=SummariesResponse)
async def get_summaries(doc_id: str):
    """
    Get the slide-by-slide summaries for display.

    Returns all summaries from Step 9.
    """
    if doc_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[doc_id].status != "completed":
        raise HTTPException(status_code=400, detail="Pipeline not yet completed")

    # Load step9 output
    output_dir = Path(settings.output_dir) / doc_id
    summaries_path = output_dir / f"{doc_id}_step9_summaries.json"

    if not summaries_path.exists():
        raise HTTPException(status_code=404, detail="Summaries not found")

    with open(summaries_path, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)

    summaries = [SlideSummary(**s) for s in summaries_data]

    return SummariesResponse(doc_id=doc_id, summaries=summaries)


@app.get("/export/{doc_id}")
async def export_docx(doc_id: str):
    """
    Download the Word document with slide summaries.

    Returns the .docx file generated in Step 9.
    """
    if doc_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[doc_id].status != "completed":
        raise HTTPException(status_code=400, detail="Pipeline not yet completed")

    # Find docx file
    output_dir = Path(settings.output_dir) / doc_id
    docx_path = output_dir / f"{doc_id}_summaries.docx"

    if not docx_path.exists():
        raise HTTPException(status_code=404, detail="Word document not found")

    return FileResponse(
        path=str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{doc_id}_summaries.docx",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
