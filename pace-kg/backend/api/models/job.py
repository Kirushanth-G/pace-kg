from dataclasses import dataclass


@dataclass
class JobStatus:
    job_id: str
    doc_id: str
    status: str          # queued | parsing | preprocessing | processing | expanding | aggregating | complete | failed
    slides_total: int = 0
    slides_completed: int = 0
    current_step: str = ""
    error: str = ""
