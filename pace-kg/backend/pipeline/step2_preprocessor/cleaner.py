"""Step 2 — Markdown Preprocessor.

Raw markdown per slide → clean typed content tree (SlideContent).

Four stages (in order):
  1. Structural parsing  — split text into typed buckets.
  2. Noise removal       — discard lines matching known noise patterns.
  3. Cross-slide filter  — remove blocks that appear on >50% of slides.
  4. Assembly            — join buckets into clean_text; expose heading_phrases.

Public API:
    preprocess_slides(slides: list[SlideMarkdown]) -> list[SlideContent]
        Run all four stages. Must be called with ALL slides at once so the
        cross-slide repetition filter has the full corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import Counter

from pipeline.step1_marker.parser import SlideMarkdown

# ---------------------------------------------------------------------------
# Noise patterns (case-insensitive, matched against full stripped text block)
# ---------------------------------------------------------------------------
NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^©.*", re.IGNORECASE),
    re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"^\s*\[\d+\]"),
]


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    return any(p.fullmatch(stripped) or p.match(stripped) for p in NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class SlideContent:
    slide_id: str
    page_number: int
    doc_id: str

    # Typed buckets (post-Stage-2 noise removal; pre-Stage-3 cross-slide filter)
    headings: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    table_cells: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    body_text: list[str] = field(default_factory=list)

    # Derived (set by Stage 4)
    heading_phrases: list[str] = field(default_factory=list)
    clean_text: str = ""


# ---------------------------------------------------------------------------
# Stage 1 — Structural parsing
# ---------------------------------------------------------------------------
def _parse_buckets(raw_markdown: str) -> dict[str, list[str]]:
    """Split raw markdown lines into typed buckets."""
    buckets: dict[str, list[str]] = {
        "headings": [],
        "bullets": [],
        "table_cells": [],
        "captions": [],
        "body_text": [],
    }

    for line in raw_markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(("#",)):
            # Heading: strip all leading # and whitespace
            text = re.sub(r"^#+\s*", "", stripped)
            if text:
                buckets["headings"].append(text)

        elif stripped.startswith(("-", "*")):
            text = re.sub(r"^[-*]\s*", "", stripped)
            if text:
                buckets["bullets"].append(text)

        elif stripped.startswith("|") and stripped.endswith("|"):
            # Table row — split on | and strip each cell
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            # Skip separator rows like |---|---|
            if all(re.fullmatch(r"[-:]+", c) for c in cells):
                continue
            buckets["table_cells"].extend(cells)

        elif stripped.startswith(">"):
            text = re.sub(r"^>\s*", "", stripped)
            if text:
                buckets["captions"].append(text)

        else:
            buckets["body_text"].append(stripped)

    return buckets


# ---------------------------------------------------------------------------
# Stage 2 — Per-line noise removal
# ---------------------------------------------------------------------------
def _remove_noise(buckets: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        key: [t for t in texts if not _is_noise(t)]
        for key, texts in buckets.items()
    }


# ---------------------------------------------------------------------------
# Stage 3 — Cross-slide repetition filter
# ---------------------------------------------------------------------------
def _cross_slide_filter(all_contents: list[SlideContent]) -> None:
    """Remove text blocks that appear on more than 50% of slides (in-place).

    Operates across all bucket types simultaneously.
    """
    n_slides = len(all_contents)
    if n_slides < 2:
        return

    threshold = n_slides * 0.5

    # Count how many slides each unique text block appears on
    block_slide_count: Counter[str] = Counter()
    for sc in all_contents:
        seen_on_this_slide: set[str] = set()
        for bucket in (sc.headings, sc.bullets, sc.table_cells, sc.captions, sc.body_text):
            for text in bucket:
                if text not in seen_on_this_slide:
                    block_slide_count[text] += 1
                    seen_on_this_slide.add(text)

    noise_blocks: set[str] = {
        text for text, count in block_slide_count.items() if count > threshold
    }

    if not noise_blocks:
        return

    for sc in all_contents:
        sc.headings    = [t for t in sc.headings    if t not in noise_blocks]
        sc.bullets     = [t for t in sc.bullets     if t not in noise_blocks]
        sc.table_cells = [t for t in sc.table_cells if t not in noise_blocks]
        sc.captions    = [t for t in sc.captions    if t not in noise_blocks]
        sc.body_text   = [t for t in sc.body_text   if t not in noise_blocks]


# ---------------------------------------------------------------------------
# Stage 4 — Assemble clean_text
# ---------------------------------------------------------------------------
def _assemble(sc: SlideContent) -> None:
    sc.heading_phrases = list(sc.headings)  # store separately for Step 3 boost
    all_parts = sc.headings + sc.body_text + sc.bullets + sc.table_cells + sc.captions
    sc.clean_text = " ".join(all_parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def preprocess_slides(slides: list[SlideMarkdown]) -> list[SlideContent]:
    """Run all 4 preprocessing stages on the full slide list.

    Must receive ALL slides at once — Stage 3 requires the complete corpus.

    Args:
        slides: Output from step1 parser.

    Returns:
        List of SlideContent in the same order as input.
    """
    contents: list[SlideContent] = []

    # Stages 1 & 2: per-slide
    for slide in slides:
        buckets = _parse_buckets(slide.raw_markdown)
        buckets = _remove_noise(buckets)
        sc = SlideContent(
            slide_id=slide.slide_id,
            page_number=slide.page_number,
            doc_id=slide.doc_id,
            headings=buckets["headings"],
            bullets=buckets["bullets"],
            table_cells=buckets["table_cells"],
            captions=buckets["captions"],
            body_text=buckets["body_text"],
        )
        contents.append(sc)

    # Stage 3: cross-slide filter (needs all slides)
    _cross_slide_filter(contents)

    # Stage 4: assemble clean_text per slide
    for sc in contents:
        _assemble(sc)

    return contents
