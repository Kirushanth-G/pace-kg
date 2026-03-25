"""Step 4 — Triple Validator (3-layer hallucination prevention).

ALL THREE LAYERS MUST PASS.  Never skip any layer.

Layer 1 — Anchor constraint
    subject and object must both appear in the keyphrase list (lowercase match).
    subject != object.
    relation must be one of the 8 allowed relation types.

Layer 2 — Evidence verification
    evidence must be non-empty and semantically similar to the slide text
    (cosine similarity ≥ EVIDENCE_SIMILARITY_THRESHOLD using SBERT).
    This catches fabricated or paraphrased "evidence" sentences.

Layer 3 — Confidence threshold
    The LLM-reported confidence must be ≥ TRIPLE_CONFIDENCE_THRESHOLD.

Public API:
    TripleValidator(config).validate(raw, keyphrases, slide_text, slide_id, doc_id)
        Returns a Triple dataclass on success, None if any layer fails.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import util

from api.models.concept import Keyphrase
from api.models.triple import Triple
from core.config import Settings
from core.embeddings import get_sbert

logger = logging.getLogger(__name__)

# The 8 allowed relation types (must match CLAUDE.md exactly)
_VALID_RELATIONS: frozenset[str] = frozenset(
    {
        "isPrerequisiteOf",
        "isDefinedAs",
        "isExampleOf",
        "contrastedWith",
        "appliedIn",
        "isPartOf",
        "causeOf",
        "isGeneralizationOf",
    }
)


class TripleValidator:
    """Validates a single raw triple dict through 3 layers.

    Instantiate once per pipeline run — it references the SBERT singleton
    which is already loaded; no additional cost per call.

    Args:
        config: Settings with threshold values.
    """

    def __init__(self, config: Settings) -> None:
        self.config = config

    def validate(
        self,
        raw: dict,
        keyphrases: List[Keyphrase],
        slide_text: str,
        slide_id: str,
        doc_id: str,
    ) -> Triple | None:
        """Run all 3 validation layers on a raw triple dict.

        Args:
            raw:        Dict from TripleExtractor.extract() — keys:
                        subject, relation, object, evidence, confidence.
            keyphrases: List[Keyphrase] for this slide (Layer 1 anchor set).
            slide_text: clean_text of the slide (Layer 2 evidence check).
            slide_id:   Slide identifier (stored on the returned Triple).
            doc_id:     Document identifier (stored on the returned Triple).

        Returns:
            Triple on success, None if any layer fails.
        """
        # ── Normalise fields ─────────────────────────────────────────────────
        subject = str(raw.get("subject", "")).lower().strip()
        obj = str(raw.get("object", "")).lower().strip()
        relation = str(raw.get("relation", "")).strip()
        evidence = str(raw.get("evidence", "")).strip()
        try:
            conf = float(raw.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0

        # ── Layer 1: Anchor constraint ────────────────────────────────────────
        kp_lower: set[str] = {k.phrase.lower() for k in keyphrases}

        if subject not in kp_lower:
            logger.debug(
                "[%s] L1 FAIL: subject %r not in keyphrase set.", slide_id, subject
            )
            return None

        if obj not in kp_lower:
            logger.debug("[%s] L1 FAIL: object %r not in keyphrase set.", slide_id, obj)
            return None

        if subject == obj:
            logger.debug("[%s] L1 FAIL: subject == object (%r).", slide_id, subject)
            return None

        if relation not in _VALID_RELATIONS:
            logger.debug(
                "[%s] L1 FAIL: relation %r not in allowed set.", slide_id, relation
            )
            return None

        # ── Layer 2: Evidence verification ────────────────────────────────────
        if not evidence:
            logger.debug("[%s] L2 FAIL: empty evidence string.", slide_id)
            return None

        sbert = get_sbert()
        ev_emb = sbert.encode(evidence, convert_to_tensor=True, show_progress_bar=False)
        text_emb = sbert.encode(
            slide_text, convert_to_tensor=True, show_progress_bar=False
        )
        sim = float(util.cos_sim(ev_emb, text_emb))

        if sim < self.config.evidence_similarity_threshold:
            logger.debug(
                "[%s] L2 FAIL: evidence sim=%.3f < threshold=%.3f — evidence: %.80s",
                slide_id,
                sim,
                self.config.evidence_similarity_threshold,
                evidence,
            )
            return None

        # ── Layer 3: Confidence threshold ─────────────────────────────────────
        if conf < self.config.triple_confidence_threshold:
            logger.debug(
                "[%s] L3 FAIL: confidence=%.3f < threshold=%.3f.",
                slide_id,
                conf,
                self.config.triple_confidence_threshold,
            )
            return None

        # ── All layers passed — return Triple ─────────────────────────────────
        logger.debug(
            "[%s] PASS: (%r —[%s]→ %r)  conf=%.2f  ev_sim=%.3f",
            slide_id,
            subject,
            relation,
            obj,
            conf,
            sim,
        )
        return Triple(
            subject=subject,
            relation=relation,
            object=obj,
            evidence=evidence,
            confidence=conf,
            slide_id=slide_id,
            doc_id=doc_id,
            source="extraction",
        )

    def validate_all(
        self,
        raw_dicts: list[dict],
        keyphrases: List[Keyphrase],
        slide_text: str,
        slide_id: str,
        doc_id: str,
    ) -> list[Triple]:
        """Validate a list of raw dicts, returning only the passing Triples.

        Convenience wrapper around validate() for use in the orchestrator.
        """
        triples: list[Triple] = []
        for raw in raw_dicts:
            triple = self.validate(raw, keyphrases, slide_text, slide_id, doc_id)
            if triple is not None:
                triples.append(triple)

        logger.info(
            "[%s] Validated %d/%d triples (%.0f%% pass rate).",
            slide_id,
            len(triples),
            len(raw_dicts),
            100 * len(triples) / len(raw_dicts) if raw_dicts else 0,
        )
        return triples
