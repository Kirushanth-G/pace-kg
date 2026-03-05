"""SBERT singletons — load once, reuse everywhere.

NEVER instantiate SentenceTransformer inside a loop or per-request.
Loading per slide adds 30+ seconds of overhead.

Two models are provided:
  get_sbert()    → all-mpnet-base-v2  (~420 MB) — Steps 4, 5, 6 (quality weighting)
  get_minilm()   → all-MiniLM-L6-v2  (~80 MB)  — Step 3 only (keyphrase scoring)
"""
from sentence_transformers import SentenceTransformer

_sbert_model: SentenceTransformer | None = None
_minilm_model: SentenceTransformer | None = None


def get_sbert() -> SentenceTransformer:
    """Returns the all-mpnet-base-v2 model for Steps 4/5/6."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer("all-mpnet-base-v2")
    return _sbert_model


def get_minilm() -> SentenceTransformer:
    """Returns the all-MiniLM-L6-v2 model used for Step 3 keyphrase scoring.

    ~80 MB — lighter than all-mpnet-base-v2, well-suited for short phrases.
    Corresponds to sentence-transformers/all-MiniLM-L6-v2 used in the
    SIFRankSqueezeBERT reference implementation from coursemapper-kg.
    """
    global _minilm_model
    if _minilm_model is None:
        _minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _minilm_model
