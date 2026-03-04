"""Step 1 — pymupdf4llm PDF Parser.

PDF file → list of SlideMarkdown objects (one per page).

Key behaviour:
- Uses pymupdf4llm.to_markdown(page_chunks=True) to get per-page markdown.
- Each page chunk is a dict with 'text' (markdown) and 'metadata' (includes 'page' 0-indexed).
- Caches results keyed by sha256(pdf_bytes) so identical PDFs are never re-parsed.
- slide_id is zero-padded: slide_001, slide_002, …

Memory footprint: ~50 MB (no ML models required). Runs on 8GB machines.
Text embedded inside images is NOT extracted. For image-heavy PDFs, see
pace-kg/COLAB_MARKER_GUIDE.md — run Marker on Google Colab, download the JSON,
then use load_parsed_json() here instead of parse_pdf().

Do NOT clean or modify text here — that is Step 2.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache directory alongside this file
_CACHE_DIR = Path(__file__).parent / "_cache"


@dataclass
class SlideMarkdown:
    slide_id: str       # e.g. "slide_001"
    page_number: int    # 1-based
    raw_markdown: str
    doc_id: str


def _cache_path(pdf_hash: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{pdf_hash}.json"


def _load_cache(pdf_hash: str) -> list[SlideMarkdown] | None:
    path = _cache_path(pdf_hash)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [SlideMarkdown(**item) for item in data]
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", pdf_hash, exc)
        return None


def _save_cache(pdf_hash: str, slides: list[SlideMarkdown]) -> None:
    path = _cache_path(pdf_hash)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in slides], f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", pdf_hash, exc)


def load_parsed_json(json_path: str | Path, doc_id: str) -> list[SlideMarkdown]:
    """Load pre-parsed slides from a JSON file produced by the Colab+Marker workflow.

    Use this instead of parse_pdf() when you have already run Marker on Google Colab
    (see COLAB_MARKER_GUIDE.md) and downloaded the resulting JSON file.

    Args:
        json_path: Path to the JSON file (list of SlideMarkdown dicts).
        doc_id:    Document identifier to assign (overrides stored doc_id).

    Returns:
        List of SlideMarkdown in page order.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    slides = [SlideMarkdown(**item) for item in data]
    for slide in slides:
        slide.doc_id = doc_id
    logger.info("Loaded %d pre-parsed slides from %s", len(slides), json_path.name)
    return slides


def parse_pdf(file_path: str | Path, doc_id: str) -> list[SlideMarkdown]:
    """Parse a PDF into per-slide markdown using pymupdf4llm.

    Args:
        file_path: Absolute path to the PDF file.
        doc_id:    Unique document identifier (e.g. UUID string).

    Returns:
        List of SlideMarkdown, one per page, in page order.

    Memory usage: ~50 MB (no ML models). Speed: <2s for typical lecture PDFs.

    Note: Text embedded inside images is NOT extracted. For image-heavy PDFs,
    use the Colab+Marker workflow (see pace-kg/COLAB_MARKER_GUIDE.md) to generate
    a cached JSON, then call load_parsed_json() instead.
    """
    file_path = Path(file_path)
    pdf_bytes = file_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Cache hit — skip re-parsing
    cached = _load_cache(pdf_hash)
    if cached is not None:
        logger.info("Cache hit for %s (hash=%s, %d slides)", doc_id, pdf_hash[:8], len(cached))
        for slide in cached:
            slide.doc_id = doc_id
        return cached

    logger.info("Parsing PDF %s with pymupdf4llm (hash=%s)…", file_path.name, pdf_hash[:8])

    try:
        import pymupdf4llm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm is not installed. Run: pip install pymupdf4llm"
        ) from exc

    # page_chunks=True → list of dicts, one per page:
    # [{"text": "<markdown>", "metadata": {"page": 0, "file_path": "...", ...}}, ...]
    page_chunks: list[dict] = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

    slides: list[SlideMarkdown] = []
    for chunk in page_chunks:
        page_md = chunk.get("text", "").strip()
        # 'page' in metadata is 0-indexed
        page_num = chunk.get("metadata", {}).get("page", len(slides)) + 1
        if not page_md:
            continue
        slide_id = f"slide_{page_num:03d}"
        slides.append(
            SlideMarkdown(
                slide_id=slide_id,
                page_number=page_num,
                raw_markdown=page_md,
                doc_id=doc_id,
            )
        )

    logger.info("Parsed %d slides from %s", len(slides), file_path.name)
    _save_cache(pdf_hash, slides)
    return slides
