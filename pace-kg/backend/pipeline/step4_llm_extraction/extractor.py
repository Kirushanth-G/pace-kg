"""Step 4 — LLM Triple Extraction  (FIXED v2).

Four targeted fixes applied on top of the original 3-layer validator:

  FIX-A  isPrerequisiteOf guard
         Requires an explicit prerequisite trigger phrase in the evidence.
         Rejects gerund/infinitive operation labels as subjects or objects.
         Blocks: "construction → storing a value", "to construct → to access"

  FIX-B  Slide-type classifier + context-aware prompt addendum
         Detects toc / comparison / code_example / procedural slides and
         injects a warning paragraph into the user prompt for that slide type.

  FIX-C  isDefinedAs object validator
         Rejects triples where the object is a containing structure.
         Blocks: "node isDefinedAs a linked data structure"
         Allows: "queue isDefinedAs fifo", "jdk isDefinedAs java language environment"

  FIX-D  isExampleOf coordinate-item guard
         Requires an explicit instance-signal phrase in the evidence AND
         rejects subject/object pairs that appear as parallel sibling bullets.
         Blocks: "list isExampleOf arrays", "queue isExampleOf list"
         Allows: "a list isExampleOf a collection"
"""

from __future__ import annotations

import json
import re
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

SYSTEM_PROMPT = """You are a knowledge graph construction assistant for educational materials.

STRICT RULES -- violating any rule invalidates your entire response:
1. ONLY use concepts from the KEYPHRASE LIST as subject and object. Nothing else.
2. Every triple MUST include an EXACT sentence copied from the SLIDE TEXT as evidence.
3. If no supporting sentence exists in the slide text, omit that triple entirely.
4. Use ONLY these 8 relation types. Follow the DIRECTION rules exactly:

   isPrerequisiteOf   subject must be CONCEPTUALLY UNDERSTOOD before object can be understood
                      ONLY use this when the slide explicitly states a learning dependency.
                      e.g. "pointers --[isPrerequisiteOf]--> linked list"
                      NEVER use for sequential steps, bullet-point ordering, or code examples.
                      WRONG: "construction --[isPrerequisiteOf]--> storing a value"
                             (these are code section labels, not conceptual prerequisites)
                      WRONG: "to construct an array list --[isPrerequisiteOf]--> to access an element"
                             (these are operations listed together, not a learning dependency)

   isDefinedAs        subject IS the concept being defined; object is the DEFINITION TEXT
                      e.g. "hashmap --[isDefinedAs]--> key-value store"
                      e.g. "queue --[isDefinedAs]--> fifo"
                      NEVER use a CONTAINING STRUCTURE as the object.
                      WRONG: "node --[isDefinedAs]--> a linked data structure"
                             (the linked data structure is where node lives, not its definition)
                      CORRECT would be: "node --[isPartOf]--> a linked data structure"

   isExampleOf        subject is a SPECIFIC INSTANCE of the broader concept object
                      ONLY use when one concept is explicitly stated as a type/instance of another.
                      e.g. "arraylist --[isExampleOf]--> collection"
                      NEVER use for items that appear as PARALLEL BULLETS at the same level.
                      WRONG: "list --[isExampleOf]--> arrays"
                             (list and arrays are both listed as separate parallel topics)
                      WRONG: "queue --[isExampleOf]--> list"
                             (queue and list are distinct sibling collection types)

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
   - The slide title is a TOPIC LABEL for the slide, not an educational concept
   - Do NOT use the full slide title string as a triple object
   - Use the specific concepts within the slide instead

7. Return ONLY a valid JSON array. No markdown, no explanation, no preamble.
8. If no valid triples found, return: []

CRITICAL CONSTRAINTS:
1. NO ADMINISTRATIVE ENTITIES: Do not extract relationships between names of people,
   professors, universities, departments, course codes, or dates.
2. PEDAGOGICAL ONLY: Only extract relationships that represent academic, domain-specific,
   or theoretical concepts taught in the material.
3. EMPTY FALLBACK: If the source text contains only a title slide, administrative info,
   a table of contents, or no clear academic facts, output an empty array []."""

USER_PROMPT = """KEYPHRASE LIST (subject and object MUST come from this list):
{keyphrases}
{slide_type_addendum}
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


# ── FIX-B: Slide-type classifier ──────────────────────────────────────────────

_TOC_KW = re.compile(
    r"\b(summary|outline|agenda|overview|contents?|topics?|table of contents|"
    r"learning objectives?|objectives?|what (we|you) (will|are going to))\b",
    re.IGNORECASE,
)
_VS_KW = re.compile(
    r"\bvs\.?\b|versus|\bcompar(e|ing|ison)\b|\badvantages?\b|\bdifferences?\b",
    re.IGNORECASE,
)
_CODE_KW = re.compile(
    r"\b(example|demo|code|syntax|declaration|declaring|implementation|"
    r"usage|how to use|listing|snippet|using)\b",
    re.IGNORECASE,
)
_PROC_KW = re.compile(
    r"\b(step|procedure|algorithm|method|process|how to|workflow)\b",
    re.IGNORECASE,
)

_SLIDE_TYPE_ADDENDA: dict[str, str] = {
    "toc": (
        "\nSLIDE TYPE: TABLE OF CONTENTS / SUMMARY SLIDE\n"
        "RULE: This slide lists topics at the same level. Do NOT extract isExampleOf or "
        "isPrerequisiteOf between items that are parallel bullet points. "
        "Only extract if the slide text contains an EXPLICIT hierarchical statement "
        "(e.g. 'X is a type of Y', 'X requires Y'). If uncertain, return [].\n"
    ),
    "comparison": (
        "\nSLIDE TYPE: COMPARISON SLIDE\n"
        "RULE: PREFER contrastedWith for pairs being compared side-by-side. "
        "Do NOT use isPrerequisiteOf for comparison items. "
        "isExampleOf is only valid if one item is explicitly stated as an "
        "instance of the other.\n"
    ),
    "code_example": (
        "\nSLIDE TYPE: CODE EXAMPLE / SYNTAX SLIDE\n"
        "RULE: Bullet headings like 'construction', 'storing a value', "
        "'retrieving a value' are CODE SECTION LABELS, not standalone domain concepts. "
        "Do NOT use them as triple subjects or objects. "
        "Do NOT use isPrerequisiteOf between sequential code steps. "
        "Extract only genuine conceptual relationships between named data structures "
        "or operations.\n"
    ),
    "procedural": (
        "\nSLIDE TYPE: PROCEDURAL SLIDE\n"
        "RULE: Sequential steps in a procedure are NOT isPrerequisiteOf relationships. "
        "isPrerequisiteOf means you must UNDERSTAND concept A before concept B, "
        "not that step A comes before step B. "
        "Only use isPrerequisiteOf if the slide explicitly states a conceptual "
        "dependency.\n"
    ),
    "normal": "",
}


def _classify_slide(sc: SlideContent) -> str:
    combined = " ".join(sc.headings).lower() + " " + sc.clean_text[:200].lower()
    if _TOC_KW.search(combined):
        return "toc"
    if _VS_KW.search(combined):
        return "comparison"
    if _CODE_KW.search(combined):
        return "code_example"
    if _PROC_KW.search(combined):
        return "procedural"
    return "normal"


# ── FIX-A: isPrerequisiteOf guard ─────────────────────────────────────────────

_PREREQ_TRIGGER = re.compile(
    r"\b(requires?|prerequisite|must (know|understand|learn|study)|"
    r"needed (before|to understand)|depends on|builds on|"
    r"before (you can|learning|understanding))\b",
    re.IGNORECASE,
)
_OPERATION_LABEL = re.compile(
    r"^(to\s+\w+|storing|retrieving|constructing|creating|accessing|"
    r"inserting|removing|adding|finding|searching|sorting|building|"
    r"declaring|initializing|instantiating)\b",
    re.IGNORECASE,
)


def _is_valid_prerequisite(subject: str, obj: str, evidence: str, slide_text: str) -> bool:
    combined = (evidence + " " + slide_text).lower()
    if not _PREREQ_TRIGGER.search(combined):
        return False
    if _OPERATION_LABEL.match(subject.strip()) or _OPERATION_LABEL.match(obj.strip()):
        return False
    return True


# ── FIX-C: isDefinedAs object validator ───────────────────────────────────────

_STRUCTURE_NOUNS = re.compile(
    r"\b(structure|framework|system|architecture|class|interface|"
    r"package|module|library|hierarchy|collection|list|map|set|"
    r"data structure|a linked|the java|language environment)\b",
    re.IGNORECASE,
)

_COMMON_WORDS: frozenset[str] = frozenset({
    "node", "list", "tree", "map", "set", "queue", "stack", "array", "heap",
    "link", "data", "type", "file", "loop", "step", "sort", "scan", "find",
    "read", "call", "class", "code", "test", "item", "text", "name", "size",
    "key", "val", "tag",
})


def _is_valid_defined_as(subject: str, obj: str, evidence: str) -> bool:
    obj_words = obj.strip().split()
    if len(obj_words) <= 2:
        return True
    subj_clean = subject.strip().replace(" ", "").lower()
    if len(subj_clean) <= 5 and subj_clean.isalpha() and subj_clean not in _COMMON_WORDS:
        return True
    if _STRUCTURE_NOUNS.search(obj):
        ev_lower = evidence.lower()
        subj_lower = subject.lower()
        patterns = [
            f"{subj_lower} is {obj.lower()}",
            f"{subj_lower} is a {obj.lower()}",
            f"{subj_lower} means {obj.lower()}",
            f"{subj_lower} = {obj.lower()}",
            f"{subj_lower} stands for {obj.lower()}",
        ]
        if not any(p in ev_lower for p in patterns):
            return False
    return True


# ── FIX-D: isExampleOf coordinate-item guard ──────────────────────────────────

_INSTANCE_SIGNAL = re.compile(
    r"\b(is (a|an|one type of|a type of|a kind of|a specific|an example of)|"
    r"such as|for example|e\.g\.|including|represented as|implemented as|"
    r"are (a|an|examples of|types of|instances of))\b",
    re.IGNORECASE,
)


def _build_bullet_peers(slide_text: str) -> set[str]:
    peers: set[str] = set()
    for line in slide_text.splitlines():
        stripped = line.strip()
        if re.match(r"^[•\-\*►▶◆]\s+\S", stripped) or re.match(r"^\d+\.\s+\S", stripped):
            text = re.sub(r"^[•\-\*►▶◆\d\.]+\s+", "", stripped).strip().lower()
            if text and len(text.split()) <= 6:
                peers.add(text)
    return peers


def _strip_article(s: str) -> str:
    return re.sub(r"^(a |an |the )", "", s.lower().strip()).strip()


def _is_valid_example_of(subject: str, obj: str, evidence: str, slide_text: str) -> bool:
    if not _INSTANCE_SIGNAL.search(evidence):
        return False
    peers = _build_bullet_peers(slide_text)
    subj_peer = any(_strip_article(subject) in p or p in _strip_article(subject) for p in peers)
    obj_peer = any(_strip_article(obj) in p or p in _strip_article(obj) for p in peers)
    if subj_peer and obj_peer:
        return False
    return True


# ── LLM helpers (unchanged) ───────────────────────────────────────────────────

def _invoke_with_fallback(messages: list, max_retries: int = 3) -> Optional[str]:
    for attempt in range(1, max_retries + 1):
        try:
            return str(get_llm().invoke(messages).content)
        except Exception as e:
            text = str(e).lower()
            if "429" in text or "rate" in text:
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


# ── Extended 3-layer validator ────────────────────────────────────────────────

def _validate_triple(
    raw: dict,
    kp_set: set,
    slide_text: str,
    slide_id: str,
    doc_id: str,
    confidence_threshold: float,
    evidence_similarity_threshold: float,
) -> Optional[Triple]:
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

    # Layer 2: evidence similarity
    if not evidence:
        return None
    sbert = get_sbert()
    ev_emb = sbert.encode(evidence, convert_to_tensor=True, show_progress_bar=False)
    text_emb = sbert.encode(slide_text, convert_to_tensor=True, show_progress_bar=False)
    if float(st_util.cos_sim(ev_emb, text_emb)) < evidence_similarity_threshold:
        return None

    # Layer 3: confidence
    if conf < confidence_threshold:
        return None

    # FIX-A
    if relation == "isPrerequisiteOf":
        if not _is_valid_prerequisite(subject, obj, evidence, slide_text):
            return None

    # FIX-C
    if relation == "isDefinedAs":
        if not _is_valid_defined_as(subject, obj, evidence):
            return None

    # FIX-D
    if relation == "isExampleOf":
        if not _is_valid_example_of(subject, obj, evidence, slide_text):
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


# ── Public API (same signature — drop-in replacement) ────────────────────────

def extract_triples(
    slides: List[SlideContent],
    keyphrases_by_slide: Dict[str, List[str]],
    confidence_threshold: float | None = None,
    evidence_similarity_threshold: float | None = None,
) -> List[Triple]:
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

    all_triples: List[Triple] = []
    stats: dict = {
        "slides": 0, "skipped": 0, "raw": 0, "passed": 0,
        "blocked_prereq": 0, "blocked_definedas": 0, "blocked_exampleof": 0,
    }

    for sc in slides:
        kps = keyphrases_by_slide.get(sc.slide_id, [])
        if not kps or not sc.clean_text.strip():
            stats["skipped"] += 1
            continue

        # FIX-B: inject slide-type context into the prompt
        slide_type = _classify_slide(sc)
        addendum = _SLIDE_TYPE_ADDENDA[slide_type]

        kp_list = "\n".join(f"- {kp}" for kp in kps)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(
                keyphrases=kp_list,
                slide_type_addendum=addendum,
                slide_text=sc.clean_text,
            )),
        ]

        content_raw = _invoke_with_fallback(messages)
        raw_dicts = _parse_llm_response(content_raw or "")
        stats["raw"] += len(raw_dicts)

        kp_set = {kp.lower().strip() for kp in kps}
        slide_triples: List[Triple] = []

        for rd in raw_dicts:
            rel = str(rd.get("relation", "")).strip()
            t = _validate_triple(
                rd, kp_set, sc.clean_text, sc.slide_id, sc.doc_id,
                confidence_threshold, evidence_similarity_threshold,
            )
            if t is not None:
                slide_triples.append(t)
            else:
                if rel == "isPrerequisiteOf":
                    stats["blocked_prereq"] += 1
                elif rel == "isDefinedAs":
                    stats["blocked_definedas"] += 1
                elif rel == "isExampleOf":
                    stats["blocked_exampleof"] += 1

        all_triples.extend(slide_triples)
        stats["slides"] += 1
        stats["passed"] += len(slide_triples)

        type_tag = f" [{slide_type}]" if slide_type != "normal" else ""
        print(f"  {sc.slide_id}{type_tag}: {len(raw_dicts)} raw → {len(slide_triples)} valid")

    print(f"\nStep 4 complete:")
    print(f"  Slides processed  : {stats['slides']} (skipped: {stats['skipped']})")
    print(f"  Raw candidates    : {stats['raw']}")
    if stats["raw"]:
        print(f"  Validated triples : {stats['passed']} "
              f"({100 * stats['passed'] // stats['raw']}% pass rate)")
    print(f"  Fix-layer blocks  → prereq:{stats['blocked_prereq']}  "
          f"definedas:{stats['blocked_definedas']}  exampleof:{stats['blocked_exampleof']}")

    rel_dist = Counter(t.relation for t in all_triples)
    print("\n  Relation distribution:")
    for rel, cnt in sorted(rel_dist.items(), key=lambda x: -x[1]):
        print(f"    {rel:<22} {cnt:>3}  {'#' * cnt}")

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