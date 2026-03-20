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
# Inline markdown stripping
# ---------------------------------------------------------------------------
_INLINE_MD = re.compile(
    r"(\*\*|__)(.*?)\1"          # **bold** or __bold__
    r"|(\*|_)(.*?)\3"            # *italic* or _italic_
    r"|`([^`]+)`"                # `inline code`
    r"|~~(.*?)~~"                # ~~strikethrough~~
    r"|\[([^\]]+)\]\([^)]+\)",  # [link text](url)
    re.DOTALL,
)


def _strip_inline(text: str) -> str:
    """Remove inline markdown syntax, keeping only the visible text content."""
    # Repeatedly apply until stable (handles nested markers)
    prev = None
    while prev != text:
        prev = text
        text = _INLINE_MD.sub(
            lambda m: next(g for g in m.groups() if g is not None and g not in ("**", "__", "*", "_")),
            text,
        )
    return text.strip()


# ---------------------------------------------------------------------------
# Noise patterns (case-insensitive, matched against full stripped text block)
# ---------------------------------------------------------------------------
NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),        # Beamer slide numbers: "2 / 21"
    re.compile(r"^(?:©|\(c\)|copyright\b)", re.IGNORECASE),
    re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"^\s*\[\d+\]"),
    re.compile(r"^!\[.*?\]\(.*?\)$"),
    re.compile(r"^\s*quiz\s*:", re.IGNORECASE),
    re.compile(r"^\s*\{\d+\}\s*$"),
    re.compile(r"^(?:input|output|legend|infrastructure)\b", re.IGNORECASE),
]


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    return any(p.fullmatch(stripped) or p.match(stripped) for p in NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# Code-line detector (Step 2)
# ---------------------------------------------------------------------------
# FIX S2-3: removed '<' and '>' from code_chars — they fire on inequality
# lists and prose type-parameter examples, producing false positives.
_CODE_CHARS = frozenset("{}();=[]")

# Used by _is_prose_label (FIX S2-5)
_BULLET_RE = re.compile(
    r"^\s*[-*+]\s*(?:[\u2022\u25A0-\u25FF\u2700-\u27BF]\s*)?|"
    r"^\s*[\u2022\u25A0-\u25FF\u2700-\u27BF]\s+"
)


def _is_code_line(text: str) -> bool:
    """Heuristic: return True when a line looks like source code, not prose."""

    t = text.strip()
    if not t:
        return False
    if t.startswith("```"):
        return True

    n = sum(1 for c in t if c in _CODE_CHARS)
    word_count = len(t.split())

    # Primary signal: high density of code punctuation
    if len(t) > 5 and n / len(t) > 0.08:
        return True

    # camelCase identifier  (FIX S2-4: exclude comma-separated token lists)
    if re.search(r"\b[a-z]+[A-Z][a-zA-Z]+\b", t):
        if word_count <= 5 and t.count(",") < 2:
            return True

    # snake_case identifier
    if re.search(r"\b[a-z]+_[a-z]+\b", t):
        return True

    # Comparison / assignment operators in spacing context
    if re.search(r"\s(==|!=|<=|>=|&&|\|\||:=|->|=>)\s", t):
        return True

    # Line terminators typical of C-family code
    if t.endswith((";", "{", "}")):
        return True

    # Comment prefixes
    if re.match(r"^\s*(//|#\s|--|/\*|\*\s)", t):
        return True

    # Method / function call  (FIX S2-2: n >= 3 AND word_count <= 8)
    if re.search(r"\b\w{2,}\(", t) and n >= 3 and word_count <= 8:
        return True

    # Constructor declaration
    if re.match(r"^(default\s+)?constructor\s+\w+\s*\(", t, re.IGNORECASE):
        return True

    # Python dunder method
    if re.match(r"^__\w+__\s*\(", t):
        return True

    # FIX S2-6: dot-notation access on very short lines  (e.g. scores.length)
    if re.search(r"\b[a-z]\w*\.[a-z]\w*\b", t) and word_count <= 3:
        return True

    return False


def _is_prose_label(line: str) -> bool:
    """Return True for lines in ``` fences that are actually prose bullets."""

    s = line.strip()
    is_bullet = bool(_BULLET_RE.match(s)) or s.startswith("•")
    if not is_bullet:
        return False
    inner = _BULLET_RE.sub("", s).lstrip("•").strip()
    if not inner:
        return False
    return sum(1 for c in inner if c in _CODE_CHARS) == 0


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
    code_lines: list[str] = field(default_factory=list)

    # Derived (set by Stage 4)
    heading_phrases: list[str] = field(default_factory=list)
    clean_text: str = ""
    ocr_applied: bool = False


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
        "code_lines": [],
    }

    in_fence = False
    for line in raw_markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Code fence handling: keep everything inside fences in code_lines.
        if stripped.startswith("```"):
            in_fence = not in_fence
            buckets["code_lines"].append(stripped)
            continue

        if in_fence:
            # Some slides use code fences but the contents are actually prose bullets.
            if _is_prose_label(stripped):
                buckets["bullets"].append(_strip_inline(stripped))
            else:
                buckets["code_lines"].append(stripped)
            continue

        # Route explicit code-like lines to code_lines so they don't pollute keyphrases.
        if _is_code_line(stripped):
            buckets["code_lines"].append(stripped)
            continue

        if stripped.startswith(("#",)):
            # Heading: strip all leading # and whitespace, then inline markdown
            text = _strip_inline(re.sub(r"^#+\s*", "", stripped))
            if text:
                buckets["headings"].append(text)

        elif stripped.startswith(("-", "*")) and not stripped.startswith(("**", "*a", "*b", "*c")):
            # Bullet: avoid accidentally treating bold lines as bullets
            # A true bullet must be '- text' or '* text' (with space after marker)
            m = re.match(r"^[-*]\s+(.+)", stripped)
            if m:
                text = _strip_inline(m.group(1))
                if text:
                    buckets["bullets"].append(text)
            else:
                # Treat as body text (e.g. standalone '**bold**' line)
                text = _strip_inline(stripped)
                if text:
                    buckets["body_text"].append(text)

        elif _BULLET_RE.match(stripped):
            # Beamer / Unicode geometric shape bullets (◮ ▶ • ▪ ■ etc.)
            # Strip the leading marker character(s) then treat as a bullet.
            text = _strip_inline(_BULLET_RE.sub("", stripped).lstrip("\u2022\u25A0-\u25FF\u2700-\u27BF").strip())
            if text and not _is_noise(text):
                if _is_code_line(text):
                    buckets["code_lines"].append(text)
                else:
                    buckets["bullets"].append(text)

        elif stripped.startswith("|") and stripped.endswith("|"):
            # Table row — split on | and strip each cell
            cells = [_strip_inline(c.strip()) for c in stripped.split("|") if c.strip()]
            # Skip separator rows like |---|---|
            if all(re.fullmatch(r"[-:]+", c) for c in cells):
                continue
            buckets["table_cells"].extend(c for c in cells if c)

        elif stripped.startswith(">"):
            text = _strip_inline(re.sub(r"^>\s*", "", stripped))
            if text:
                buckets["captions"].append(text)

        else:
            text = _strip_inline(stripped)
            if text:
                buckets["body_text"].append(text)

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
def load_preprocessed_json(json_path: str, doc_id: str) -> list[SlideContent]:
    """Load a Marker-preprocessed JSON (from Colab or external pipeline) as SlideContent objects.

    Step 2 has already been performed externally; this skips all 4 stages and
    maps the JSON fields directly to SlideContent. The cross-slide filter is
    NOT re-applied (assumed done upstream).

    Handles:
    - '{N}' page-number placeholders in body_text → stripped out.
    - Extra fields ('discarded', 'ocr_applied') → ignored.
    - clean_text is rebuilt from buckets if needed to match our format.

    Args:
        json_path: Path to the preprocessed JSON file.
        doc_id:    Document identifier to assign (overrides stored doc_id).

    Returns:
        List of SlideContent in page order, ready for Step 3.
    """
    import json
    import logging as _log
    _logger = _log.getLogger(__name__)

    _PAGE_REF     = re.compile(r"^\{\d+\}$")              # matches '{1}', '{12}'
    _BULLET_LEAD  = re.compile(r"^[\s◮•▶▪■►▸*\-–—]+")    # leading bullet markers

    def _strip_bullet(text: str) -> str:
        """Remove leading bullet-marker characters (◮, •, ▶, -, etc.)."""
        return _BULLET_LEAD.sub("", text).strip()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contents: list[SlideContent] = []
    for item in data:
        # Strip '{N}' placeholders from body_text
        body = [t for t in item.get("body_text", []) if not _PAGE_REF.match(t.strip())]

        # Strip leading bullet-marker characters (◮, •, ▶, etc.) from bullets
        bullets = [_strip_bullet(b) for b in item.get("bullets", []) if _strip_bullet(b)]

        sc = SlideContent(
            slide_id=item["slide_id"],
            page_number=item["page_number"],
            doc_id=doc_id,
            headings=item.get("headings", []),
            bullets=bullets,
            table_cells=item.get("table_cells", []),
            captions=item.get("captions", []),
            body_text=body,
            code_lines=item.get("code_lines", []),
            heading_phrases=item.get("heading_phrases", []),
            ocr_applied=bool(item.get("ocr_applied", False)),
        )

        # Rebuild clean_text using our assembly format
        # (ignore the stored clean_text — rebuild to drop {N} refs and match our pipeline)
        all_parts = sc.headings + sc.body_text + sc.bullets + sc.table_cells + sc.captions
        sc.clean_text = " ".join(all_parts)

        contents.append(sc)

    _logger.info("Loaded %d pre-processed slides from %s", len(contents), json_path)
    return contents


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
            code_lines=buckets["code_lines"],
        )
        contents.append(sc)

    # Stage 3: cross-slide filter (needs all slides)
    _cross_slide_filter(contents)

    # Stage 4: assemble clean_text per slide
    for sc in contents:
        _assemble(sc)

    return contents