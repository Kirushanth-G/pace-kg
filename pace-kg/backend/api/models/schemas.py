"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response after PDF upload."""

    doc_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Pipeline job status."""

    doc_id: str
    status: str  # pending, running, completed, failed
    current_step: Optional[str] = None
    progress: float  # 0.0 to 1.0
    error: Optional[str] = None


class ConceptNode(BaseModel):
    """Knowledge graph concept node."""

    name: str
    aliases: List[str]
    slide_ids: List[str]
    source_type: str
    keyphrase_score: float
    final_weight: float
    doc_id: Optional[str] = None  # Optional - injected from URL param if missing
    needs_review: bool = False


class Edge(BaseModel):
    """Knowledge graph edge."""

    subject: str
    relation: str
    object: str
    evidence: Optional[str] = None
    confidence: float
    slide_id: str
    doc_id: Optional[str] = None  # Optional - injected from URL param if missing
    source: str  # extraction or expansion


class GraphResponse(BaseModel):
    """Full knowledge graph for frontend rendering."""

    doc_id: str
    nodes: List[ConceptNode]
    edges: List[Edge]


class SlideSummary(BaseModel):
    """Per-slide summary for revision notes."""

    slide_id: str
    page_number: int
    heading: str
    summary: str
    key_terms: List[str]
    doc_id: Optional[str] = None  # Optional - injected from URL param if missing


class SummariesResponse(BaseModel):
    """Collection of slide summaries."""

    doc_id: str
    summaries: List[SlideSummary]
