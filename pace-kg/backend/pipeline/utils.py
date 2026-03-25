"""Shared pipeline utilities — importable by both cleaner.py and extractor.py.

Kept in a separate module to avoid circular imports between Step 2 and Step 3.
"""
from __future__ import annotations

import re


def is_code_line(text: str) -> bool:
    """Detect code lines using structural signals that work across ALL languages.

    Detects Java, Python, C++, SQL, R, JavaScript, MATLAB, pseudocode, etc.
    Does NOT check for language-specific keywords — uses punctuation density,
    naming conventions, and operator patterns instead.

    Called by:
      - cleaner.py (Step 2) to route body_text lines into code_lines bucket
      - extractor.py (Step 3) as a safety-net filter when building sifrank_text

    Returns True if the line is very likely code / pseudocode.
    """
    t = text.strip()
    if not t:
        return False

    # Signal 1: code fence marker (Marker output wraps code blocks)
    if t.startswith("```"):
        return True

    # Signal 2: high density of code punctuation {}();=<>[]
    code_chars = set("{}();=<>[]")
    code_char_count = sum(1 for c in t if c in code_chars)
    if len(t) > 5 and code_char_count / len(t) > 0.10:
        return True

    # Signal 3: camelCase token inside the line  e.g. HashMap, LinkedList, getKey
    # Requires a word that starts with lowercase and has an internal uppercase
    # OR a CapitalCase word followed immediately by a lowercase+uppercase sequence
    if re.search(r"\b[a-z]+[A-Z][a-zA-Z]+\b", t):
        return True

    # Signal 4: snake_case token  e.g. initial_capacity, compute_weight
    if re.search(r"\b[a-z]+_[a-z]+\b", t):
        return True

    # Signal 5: assignment / comparison / logical operators
    if re.search(r"\s(==|!=|<=|>=|&&|\|\||:=|->|=>)\s", t):
        return True

    # Signal 6: line ends with ; or { or }  (statement terminator / block)
    if t.endswith((";", "{", "}")):
        return True

    # Signal 7: comment patterns across languages  //, #, --, /*, *
    if re.match(r"^\s*(//|#\s|--|/\*|\*\s)", t):
        return True

    # Signal 8: function/method/constructor call pattern  word( with >= 2 code chars
    # Require code_char_count >= 2 to avoid false positives on
    # natural phrases like "HashMap (a data structure)"
    if re.search(r"\b\w{2,}\(", t) and code_char_count >= 2:
        return True

    # Signal 9: constructor bullet pattern from Marker output
    # e.g. "1. Default constructor, HashMap() creates an instance..."
    # e.g. "2. Constructor HashMap(int initialCapacity) creates..."
    if re.match(r"^\d+\.\s+(default\s+)?constructor\s+\w+\s*\(", t, re.IGNORECASE):
        return True
    if re.match(r"^(default\s+)?constructor\s+\w+\s*\(", t, re.IGNORECASE):
        return True

    # Signal 10: Python dunder methods
    if re.match(r"__\w+__\s*\(", t):
        return True

    return False
