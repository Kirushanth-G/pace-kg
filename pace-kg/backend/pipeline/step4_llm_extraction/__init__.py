"""Step 4 — LLM triple extraction.

This package exposes functions to run the Groq LLM extractor and validate
triples against in-slide evidence.
"""

from .extractor import extract_triples, load_keyphrases, save_triples
