"""Step 4 — LLM Triple Extraction.

Sends (keyphrases + slide text) to Groq's Llama model and returns a raw list
of dicts ready for TripleValidator.  Automatic fallback to the smaller model
when the primary hits HTTP 429 (rate limit).

Public API:
    TripleExtractor(config).extract(keyphrases, slide_content) -> list[dict]
        Returns raw dicts (not yet validated Triples).
        The caller (orchestrator / test) should pass each dict to
        TripleValidator.validate() to get a Triple or None.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from api.models.concept import Keyphrase
from core.config import Settings
from core.llm_client import get_fallback_llm, get_llm
from pipeline.step2_preprocessor.cleaner import SlideContent
from pipeline.step4_llm_extraction.prompts import SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)

# Maximum retries when both models are rate-limited
_MAX_RETRIES = 3
_RETRY_DELAY_S = 5  # seconds between retries


def _invoke_with_fallback(messages: list, config: Settings) -> str | None:
    """Invoke the primary LLM; fall back automatically on HTTP 429.

    Returns the raw string content from the model, or None on failure.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        # Try primary
        try:
            llm: ChatGroq = get_llm()
            response = llm.invoke(messages)
            return str(response.content)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate" in exc_str.lower():
                logger.warning(
                    "Primary LLM rate-limited (attempt %d/%d) — switching to fallback.",
                    attempt,
                    _MAX_RETRIES,
                )
                # Try fallback model
                try:
                    fallback: ChatGroq = get_fallback_llm()
                    response = fallback.invoke(messages)
                    return str(response.content)
                except Exception as fallback_exc:
                    fb_str = str(fallback_exc)
                    if "429" in fb_str or "rate" in fb_str.lower():
                        logger.warning(
                            "Fallback LLM also rate-limited. Waiting %ds …",
                            _RETRY_DELAY_S,
                        )
                        time.sleep(_RETRY_DELAY_S)
                        continue
                    logger.error("Fallback LLM error: %s", fallback_exc)
                    return None
            else:
                logger.error("Primary LLM error: %s", exc)
                return None
    logger.error("All %d LLM attempts exhausted.", _MAX_RETRIES)
    return None


def _parse_response(content: str) -> list[dict]:
    """Parse JSON from LLM response content.

    The model is configured for JSON mode, so content should be either:
      - A JSON array directly: [{ ... }, ...]
      - A JSON object wrapping an array: {"triples": [...]} or similar
    Returns an empty list on any parse failure.
    """
    if not content:
        return []
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s — content prefix: %.200s", e, content)
        return []

    if isinstance(raw, list):
        return raw

    # Groq JSON-mode sometimes wraps in an object; find the first list value
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                return v

    logger.warning("Unexpected LLM response shape: %s", type(raw).__name__)
    return []


class TripleExtractor:
    """Extracts raw triple dicts from a single slide using the Groq LLM.

    Usage:
        extractor = TripleExtractor(config)
        raw_dicts = extractor.extract(keyphrases, slide_content)
        # Then pass each dict to TripleValidator.validate()
    """

    def __init__(self, config: Settings) -> None:
        self.config = config

    def extract(
        self,
        keyphrases: List[Keyphrase],
        slide_content: SlideContent,
    ) -> list[dict]:
        """Call the LLM and return raw (unvalidated) triple dicts.

        Args:
            keyphrases:    List[Keyphrase] from Step 3 for this slide.
            slide_content: SlideContent from Step 2 for this slide.

        Returns:
            List of raw dicts — each may have keys:
                subject, relation, object, evidence, confidence
            Returns [] if no keyphrases, empty slide, or LLM error.
        """
        if not keyphrases:
            logger.debug(
                "[%s] No keyphrases — skipping LLM call.", slide_content.slide_id
            )
            return []

        if not slide_content.clean_text.strip():
            logger.debug(
                "[%s] Empty slide text — skipping LLM call.", slide_content.slide_id
            )
            return []

        kp_list = "\n".join(f"- {k.phrase}" for k in keyphrases)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT.format(
                    keyphrases=kp_list,
                    slide_text=slide_content.clean_text,
                )
            ),
        ]

        logger.info(
            "[%s] Sending %d keyphrases to LLM for triple extraction.",
            slide_content.slide_id,
            len(keyphrases),
        )

        content = _invoke_with_fallback(messages, self.config)
        raw_dicts = _parse_response(content or "")

        logger.info(
            "[%s] LLM returned %d raw triple candidate(s).",
            slide_content.slide_id,
            len(raw_dicts),
        )
        return raw_dicts
