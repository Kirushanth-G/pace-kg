from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    llm_primary: str = "llama-3.3-70b-versatile"
    llm_fallback: str = "llama-3.1-8b-instant"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Pipeline thresholds
    keyphrase_max_candidates: int = 30
    keyphrase_quality_threshold: float = 0.3
    weight_pruning_threshold: float = 0.192
    expansion_similarity_threshold: float = 0.65
    expansion_max_related: int = 10
    triple_confidence_threshold: float = 0.70
    # Evidence similarity threshold is intentionally lower (0.65) to match the
    # Colab pipeline fix (CLAUDE.md Step 4) where a stricter default caused
    # too many triples to be rejected.
    evidence_similarity_threshold: float = 0.65
    conflict_merge_threshold: float = 0.92

    # Storage
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
