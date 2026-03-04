from dataclasses import dataclass, field
from typing import Literal, List

RelationType = Literal[
    "isPrerequisiteOf",
    "isDefinedAs",
    "isExampleOf",
    "contrastedWith",
    "appliedIn",
    "isPartOf",
    "causeOf",
    "isGeneralizationOf",
]


@dataclass
class Triple:
    subject: str        # lowercase, stripped
    relation: str       # one of RelationType
    object: str         # lowercase, stripped
    evidence: str       # exact sentence from PDF — NEVER paraphrased
    confidence: float   # 0.0 to 1.0
    slide_id: str
    doc_id: str
    source: str = "extraction"


@dataclass
class ExpansionEdge:
    subject: str
    object: str
    relation: str = "relatedConcept"
    source: str = "expansion"
    confidence: float = 0.0
    slide_id: str = ""
    doc_id: str = ""
