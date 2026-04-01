import os
from typing import Dict, Tuple
from pydantic_settings import BaseSettings


# Step 5 relation role boosts (constant dict)
RELATION_ROLE_BOOSTS: Dict[Tuple[str, str], float] = {
    ("isDefinedAs", "object"): +0.15,
    ("isPrerequisiteOf", "subject"): +0.10,
    ("isGeneralizationOf", "object"): +0.10,
    ("contrastedWith", "subject"): +0.05,
    ("contrastedWith", "object"): +0.05,
    ("causeOf", "subject"): +0.05,
    ("isExampleOf", "subject"): -0.05,
}

# Step 5 source type boosts (constant dict)
SOURCE_TYPE_BOOSTS: Dict[str, float] = {
    "heading": +0.10,
    "bullet": +0.05,
    "table": +0.05,
    "body": +0.00,
    "caption": -0.05,
    "injected": -0.05,
}


class Settings(BaseSettings):
    # LLM - read from environment
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_primary: str = "llama-3.3-70b-versatile"
    llm_fallback: str = "llama-3.1-8b-instant"
    llm_summary: str = "llama-3.1-8b-instant"  # Step 9 uses cheap model

    # Neo4j - read from environment
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Pipeline thresholds (from CLAUDE.md Section 6)
    page_sep: str = "\n\n<<<MARKER_PAGE_BREAK>>>\n\n"
    keyphrase_max: int = 25
    gliner_threshold: float = 0.35
    dedup_sim_threshold: float = 0.85
    heading_boost: float = 0.15
    llm_fallback_score: float = 0.40
    triple_confidence_threshold: float = 0.70
    evidence_similarity_threshold: float = 0.65
    evidence_similarity_threshold_llm: float = 0.50

    # Step 5 - Concept weighting
    w_evidence: float = 0.50
    w_slide: float = 0.30
    w_doc: float = 0.20
    weight_threshold: float = 0.192
    merge_sim_threshold: float = 0.92
    review_sim_threshold: float = 0.75

    # Step 6 - Expansion
    sbert_sim_threshold: float = 0.65
    doc_weight_threshold: float = 0.70
    max_candidates_per_concept: int = 10

    # Storage
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    max_upload_size_mb: int = 50

    model_config = {"extra": "ignore", "env_file": ".env", "case_sensitive": False}


settings = Settings()
