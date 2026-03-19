"""
Step 3 — Keyphrase Extraction (same as pace_kg.py)

Loads preprocessed slides from JSON and extracts keyphrases using GLiNER.
Uses the exact same logic as pace_kg.py Step 3 cell.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict

import spacy
from gliner import GLiNER
from sentence_transformers import SentenceTransformer, util as st_util

# -- Config -------------------------------------------------------------------
GLINER_MODEL           = "urchade/gliner_large-v2.1"
GLINER_THRESHOLD       = 0.35
DEDUP_SIM_THRESHOLD    = 0.85
KEYPHRASE_MAX          = 25
HEADING_BOOST          = 0.15

GLINER_LABELS = [
    "Academic Concept",
    "Theoretical Principle",
    "Technical Term",
    "Process or Method",
    "System or Framework",
    "Formula or Equation",
]
ENTITY_LABELS = GLINER_LABELS

# -- Data models ---------------------------------------------------------------
@dataclass
class SlideContent:
    slide_id:        str
    page_number:     int
    doc_id:          str
    headings:        List[str]
    bullets:         List[str]
    table_cells:     List[str]
    captions:        List[str]
    body_text:       List[str]
    heading_phrases: List[str]
    clean_text:      str

@dataclass
class Keyphrase:
    phrase:      str
    score:       float
    source_type: str
    slide_id:    str
    doc_id:      str
    appears_in:  str

# -- Load models ---------------------------------------------------------------
print("Loading GLiNER large-v2.1 on GPU ...")
gliner = GLiNER.from_pretrained(GLINER_MODEL)
print("GLiNER ready.")

print("Loading spaCy en_core_web_sm (sentence segmentation) ...")
nlp = spacy.load("en_core_web_sm")

print("Loading all-MiniLM-L6-v2 (deduplication) ...")
minilm = SentenceTransformer("all-MiniLM-L6-v2")
print("All models ready.")

# -- Helpers ------------------------------------------------------------------
def _assign_source_type(phrase: str, sc) -> str:
    pl = phrase.lower()
    for h in sc.headings:
        if pl in h.lower(): return "heading"
    for b in sc.bullets:
        if pl in b.lower(): return "bullet"
    for t in sc.table_cells:
        if pl in t.lower(): return "table"
    for c in sc.captions:
        if pl in c.lower(): return "caption"
    return "body"

def _find_sentence(phrase: str, clean_text: str) -> str:
    doc = nlp(clean_text)
    pl  = phrase.lower()
    for sent in doc.sents:
        if pl in sent.text.lower():
            return sent.text.strip()
    return clean_text[:200]

def _deduplicate(kps: List[Keyphrase]) -> List[Keyphrase]:
    if len(kps) <= 1:
        return kps
    sorted_kps = sorted(kps, key=lambda k: k.score, reverse=True)
    kept: List[Keyphrase] = []
    for kp in sorted_kps:
        emb = minilm.encode(kp.phrase, convert_to_tensor=True)
        is_dup = any(
            float(st_util.cos_sim(
                emb,
                minilm.encode(k.phrase, convert_to_tensor=True)
            )) >= DEDUP_SIM_THRESHOLD
            for k in kept
        )
        if not is_dup:
            kept.append(kp)
    return kept

# -- Main extraction ----------------------------------------------------------
def extract_keyphrases(sc) -> List[Keyphrase]:
    if not sc.clean_text.strip():
        return []

    heading_text = " ".join(sc.headings)
    rest_text    = " ".join(
        sc.body_text + sc.bullets + sc.table_cells + sc.captions
    )
    extract_text = (heading_text + ". " + rest_text).strip() \
                   if heading_text and rest_text else (heading_text or rest_text)

    if not extract_text.strip():
        return []

    try:
        entities = gliner.predict_entities(
            extract_text,
            ENTITY_LABELS,
            threshold=GLINER_THRESHOLD,
        )
    except Exception as e:
        print(f"    GLiNER error on {sc.slide_id}: {e}")
        return []

    if not entities:
        return []

    best: Dict[str, float] = {}
    for ent in entities:
        phrase = ent["text"].lower().strip()
        score  = float(ent["score"])
        if len(phrase) < 3:
            continue
        if phrase not in best or score > best[phrase]:
            best[phrase] = score

    kps: List[Keyphrase] = []
    for phrase, score in best.items():
        src   = _assign_source_type(phrase, sc)
        final = min(score + HEADING_BOOST, 1.0) if src == "heading" else score
        app   = _find_sentence(phrase, sc.clean_text)
        kps.append(Keyphrase(
            phrase=phrase,
            score=round(final, 4),
            source_type=src,
            slide_id=sc.slide_id,
            doc_id=sc.doc_id,
            appears_in=app,
        ))

    kps = _deduplicate(kps)
    kps.sort(key=lambda k: k.score, reverse=True)
    return kps[:KEYPHRASE_MAX]

def is_pedagogical_slide(sc) -> bool:
    word_count = len(sc.clean_text.split())
    if sc.page_number == 1 and word_count < 40:
        return False
    if word_count < 8:
        return False
    return True

# -- Main script ---------------------------------------------------------------
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", default="CS1050-L03_preprocessed.json")
parser.add_argument("--doc-id", default=None)
args = parser.parse_args()

inp_path = Path(args.input)
if not inp_path.exists():
    print(f"ERROR: {inp_path} not found")
    sys.exit(1)

# Load preprocessed JSON
print(f"\nLoading preprocessed slides from {inp_path}...")
with open(inp_path, "r", encoding="utf-8") as f:
    data = json.load(f)

content_slides = [SlideContent(**item) for item in data]
print(f"Loaded {len(content_slides)} slides\n")

# Extract keyphrases
print("Extracting keyphrases with GLiNER large from pedagogical slides ...")
keyphrases_by_slide: Dict[str, List[Keyphrase]] = {}
label_dist: Dict[str, int] = {}

for i, sc in enumerate(content_slides, 1):
    if not is_pedagogical_slide(sc):
        print(f"  [{i:2d}/{len(content_slides)}] {sc.slide_id}: SKIPPED (Administrative/Low-content, {len(sc.clean_text.split())} words)")
        keyphrases_by_slide[sc.slide_id] = []
        continue

    kps = extract_keyphrases(sc)
    keyphrases_by_slide[sc.slide_id] = kps

    raw_text = " ".join(sc.headings + sc.body_text + sc.bullets
                        + sc.table_cells + sc.captions).strip()
    if raw_text:
        try:
            raw_ents = gliner.predict_entities(
                raw_text, ENTITY_LABELS, threshold=GLINER_THRESHOLD
            )
            for e in raw_ents:
                label_dist[e["label"]] = label_dist.get(e["label"], 0) + 1
        except Exception:
            pass

    print(f"  [{i:2d}/{len(content_slides)}] {sc.slide_id}: {len(kps)} keyphrases")
    if kps:
        top3 = ", ".join(f"{k.phrase} ({k.score:.2f})" for k in kps[:3])
        print(f"             top-3: {top3}")

total_kps = sum(len(v) for v in keyphrases_by_slide.values())
print(f"\nStep 3 complete : {total_kps} keyphrases across {len(content_slides)} slides")
print(f"Avg per slide   : {total_kps/len(content_slides):.1f}")
print()
print("Entity label distribution (raw, before dedup):")
for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
    print(f"  {lbl:<38} {cnt}")

# Determine output filename
doc_id = args.doc_id or (inp_path.stem.replace("_preprocessed", "") if "_preprocessed" in inp_path.stem else inp_path.stem)
STEP3_FILE = f"{doc_id}_keyphrases.json"

step3_out  = {sid: [asdict(k) for k in kps]
              for sid, kps in keyphrases_by_slide.items()}
with open(STEP3_FILE, "w", encoding="utf-8") as f:
    json.dump(step3_out, f, ensure_ascii=False, indent=2)
print(f"\nSaved  : {STEP3_FILE} ({Path(STEP3_FILE).stat().st_size:,} bytes)")

print(f"\n{'slide_id':<12}  {'#kps':>4}  top-5 keyphrases (phrase — score — source)")
print("-" * 100)
for sc in content_slides:
    kps = keyphrases_by_slide.get(sc.slide_id, [])
    top5 = ", ".join(
        f"{k.phrase} ({k.score:.2f}/{k.source_type[0]})"
        for k in kps[:5]
    )
    print(f"{sc.slide_id:<12}  {len(kps):>4}  {top5}")

print(f"\n✓ Step 3 complete. Output: {STEP3_FILE}")
