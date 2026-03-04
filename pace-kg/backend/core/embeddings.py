"""SBERT singleton — load once, reuse everywhere.
NEVER instantiate SentenceTransformer inside a loop or per-request.
Loading per slide adds 30+ seconds of overhead.
"""
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_sbert() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model
