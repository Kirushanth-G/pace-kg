"""Step 4 — LLM Triple Extraction.

This step mirrors the logic in CLAUDE.md / pace_kg.py (Colab notebook) but runs
in a standard Python environment.

Input:
  - SlideContent objects (preprocessed step 2 output)
  - Keyphrase list per slide (step 3 output)

Output:
  - List[Triple] objects (extraction triples).

This module is intentionally self-contained so it can be used from command-line
scripts or from the API backend.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional

from sentence_transformers import util as st_util
from langchain_core.messages import HumanMessage, SystemMessage

from api.models.triple import Triple
from core.config import settings
from core.embeddings import get_sbert
from core.llm_client import get_fallback_llm, get_llm
from pipeline.step2_preprocessor.cleaner import SlideContent


VALID_RELATIONS = frozenset({
    "isPrerequisiteOf",
    "isDefinedAs",
    "isExampleOf",
    "contrastedWith",
    "appliedIn",
    "isPartOf",
    "causeOf",
    "isGeneralizationOf",
})

# Copies the system/user prompts from CLAUDE.md (Step 4) to enforce consistency.
SYSTEM_PROMPT = """You are a knowledge graph construction assistant for educational materials.

STRICT RULES -- violating any rule invalidates your entire response:
1. ONLY use concepts from the KEYPHRASE LIST as subject and object. Nothing else.
2. Every triple MUST include an EXACT sentence copied from the SLIDE TEXT as evidence.
3. If no supporting sentence exists in the slide text, omit that triple entirely.
4. Use ONLY these 8 relation types. Follow the DIRECTION rules exactly:

   isPrerequisiteOf   subject must be understood BEFORE object can be understood
                      e.g. "pointers --[isPrerequisiteOf]--> linked list"

   isDefinedAs        subject IS the concept being defined; object is the definition
                      e.g. "hashmap --[isDefinedAs]--> key-value store"
                      NEVER reverse: the concept comes first, the definition second

   isExampleOf        subject is a SPECIFIC INSTANCE of the broader concept object
                      e.g. "arraylist --[isExampleOf]--> collection"
                      e.g. "linkedlist --[isExampleOf]--> linked list data structure"

   contrastedWith     subject AND object are EXPLICITLY compared or contrasted
                      e.g. "hashmap --[contrastedWith]--> linkedlist"
                      USE THIS when the slide title contains "vs", "comparison", "advantages"
                      or when the slide text uses words like "unlike", "whereas", "compared to"

   appliedIn          subject concept IS USED IN or runs INSIDE object
                      e.g. "hash function --[appliedIn]--> hashmap"
                      e.g. "hashmap --[appliedIn]--> jvm"

   isPartOf           subject is a PHYSICAL or STRUCTURAL COMPONENT of object
                      e.g. "load factor --[isPartOf]--> hashmap parameters"
                      DO NOT use for: taxonomy membership, conceptual groupings,
                      or when one concept IS AN IMPLEMENTATION of another

   causeOf            subject DIRECTLY CAUSES or leads to object as a result
                      e.g. "rehashing --[causeOf]--> capacity increase"
                      e.g. "capacity x load factor --[causeOf]--> rehashing threshold"

   isGeneralizationOf subject is a BROADER CATEGORY that contains object as a member
                      e.g. "collection --[isGeneralizationOf]--> hashmap"

5. DIVERSITY REQUIREMENT:
   - Consider ALL 8 relation types before choosing one
   - Do NOT use isPartOf for more than 40% of your triples in this response
   - If the slide title contains "vs", "comparison", or "advantages", you MUST
     produce at least one contrastedWith triple
   - If the slide text contains "is an implementation of" or "is defined as",
     use isDefinedAs or isExampleOf -- not isPartOf
   - If the slide text contains "causes", "results in", or "leads to",
     use causeOf -- not isPartOf

6. SLIDE HEADING RULE:
   - The slide title (e.g. "Java pre- and post- Collections", "HashMap Class")
     is a TOPIC LABEL for the slide, not an educational concept
   - Do NOT use the full slide title string as a triple object
   - Use the specific concepts within the slide instead

7. Return ONLY a valid JSON array. No markdown, no explanation, no preamble.
8. If no valid triples found, return: []

CRITICAL CONSTRAINTS:
1. NO ADMINISTRATIVE ENTITIES: Do not extract relationships between names of people, professors, universities, departments, course codes, or dates.
2. PEDAGOGICAL ONLY: Only extract relationships that represent academic, domain-specific, or theoretical concepts taught in the material.
3. EMPTY FALLBACK: If the source text contains only a title slide, administrative info, a table of contents, or no clear academic facts, output an empty array []."""

USER_PROMPT = """KEYPHRASE LIST (subject and object MUST come from this list):
{keyphrases}

SLIDE TEXT:
{slide_text}

Extract all valid triples as JSON array. Each item:
{{
  "subject": "exact phrase from keyphrase list",
  "relation": "one of the 8 relation types",
  "object": "exact phrase from keyphrase list (different from subject)",
  "evidence": "exact sentence copied from slide text above",
  "confidence": 0.0 to 1.0
}}"""


def _invoke_with_fallback(messages: list, max_retries: int = 3) -> Optional[str]:
    """Invoke Groq LLM, falling back on rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return str(get_llm().invoke(messages).content)
        except Exception as e:
            text = str(e).lower()
            if "429" in text or "rate" in text:
                # Rate limited: try fallback, with a small backoff.
                try:
                    return str(get_fallback_llm().invoke(messages).content)
                except Exception as e2:
                    text2 = str(e2).lower()
                    if "429" in text2 or "rate" in text2:
                        time.sleep(5)
                        continue
                    raise
            raise
    return None


def _parse_llm_response(content: str) -> List[dict]:
    if not content:
        return []
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                return v
    return []


def _validate_triple(
    raw: dict,
    kp_set: set,
    slide_text: str,
    slide_id: str,
    doc_id: str,
    confidence_threshold: float,
    evidence_similarity_threshold: float,
) -> Optional[Triple]:
    """Perform the 3 validation layers described in CLAUDE.md Step 4."""
    subject = str(raw.get("subject", "")).lower().strip()
    obj = str(raw.get("object", "")).lower().strip()
    relation = str(raw.get("relation", "")).strip()
    evidence = str(raw.get("evidence", "")).strip()
    try:
        conf = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0

    # Layer 1: anchor + schema
    if subject not in kp_set or obj not in kp_set:
        return None
    if subject == obj:
        return None
    if relation not in VALID_RELATIONS:
        return None

    # Layer 2: evidence verification
    if not evidence:
        return None

    sbert = get_sbert()
    ev_emb = sbert.encode(evidence, convert_to_tensor=True, show_progress_bar=False)
    text_emb = sbert.encode(slide_text, convert_to_tensor=True, show_progress_bar=False)
    sim = float(st_util.cos_sim(ev_emb, text_emb))
    if sim < evidence_similarity_threshold:
        return None

    # Layer 3: confidence
    if conf < confidence_threshold:
        return None

    return Triple(
        subject=subject,
        relation=relation,
        object=obj,
        evidence=evidence,
        confidence=conf,
        slide_id=slide_id,
        doc_id=doc_id,
    )


def extract_triples(
    slides: List[SlideContent],
    keyphrases_by_slide: Dict[str, List[str]],
    confidence_threshold: float | None = None,
    evidence_similarity_threshold: float | None = None,
) -> List[Triple]:
    """Extract validated triples for all slides.

    Args:
        slides: SlideContent objects from Step 2.
        keyphrases_by_slide: Mapping slide_id -> list of keyphrase strings.
        confidence_threshold: Override for triple confidence.
        evidence_similarity_threshold: Override for evidence similarity.

    Returns:
        List of validated Triple objects.
    """

    confidence_threshold = (
        settings.triple_confidence_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    evidence_similarity_threshold = (
        settings.evidence_similarity_threshold
        if evidence_similarity_threshold is None
        else evidence_similarity_threshold
    )

    sbert = get_sbert()  # Load once

    all_triples: List[Triple] = []
    stats = {"slides": 0, "skipped": 0, "raw": 0, "passed": 0}

    for sc in slides:
        kps = keyphrases_by_slide.get(sc.slide_id, [])
        if not kps or not sc.clean_text.strip():
            stats["skipped"] += 1
            continue

        kp_list = "\n".join(f"- {kp}" for kp in kps)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(keyphrases=kp_list, slide_text=sc.clean_text)),
        ]

        content_raw = _invoke_with_fallback(messages)
        raw_dicts = _parse_llm_response(content_raw or "")
        stats["raw"] += len(raw_dicts)

        kp_set = {kp.lower().strip() for kp in kps}
        slide_triples: List[Triple] = []
        for rd in raw_dicts:
            t = _validate_triple(
                rd,
                kp_set,
                sc.clean_text,
                sc.slide_id,
                sc.doc_id,
                confidence_threshold,
                evidence_similarity_threshold,
            )
            if t is not None:
                slide_triples.append(t)

        all_triples.extend(slide_triples)
        stats["slides"] += 1
        stats["passed"] += len(slide_triples)

    return all_triples


def save_triples(triples: List[Triple], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in triples], f, ensure_ascii=False, indent=2)


def load_keyphrases(path: str) -> Dict[str, List[str]]:
    """Load keyphrases JSON into a mapping slide_id -> list[str]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[str]] = {}
    for slide_id, items in data.items():
        if not isinstance(items, list):
            continue
        phrases = []
        for item in items:
            if not isinstance(item, dict):
                continue
            phrase = item.get("phrase")
            if not isinstance(phrase, str):
                continue
            phrases.append(phrase.strip().lower())
        out[slide_id] = phrases
    return out
