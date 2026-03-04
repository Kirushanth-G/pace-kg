from dataclasses import dataclass, field
from typing import List

from api.models.triple import Triple


@dataclass
class Keyphrase:
    phrase: str          # lowercase, stripped
    score: float
    source_type: str     # heading | body | bullet | table | caption
    slide_id: str
    doc_id: str
    appears_in: str      # sentence containing this phrase


@dataclass
class WeightedConcept:
    name: str
    final_weight: float
    slide_id: str
    doc_id: str
    source_type: str
    keyphrase_score: float
    triples: List[Triple] = field(default_factory=list)
