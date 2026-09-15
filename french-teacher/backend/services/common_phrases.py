"""
A general-purpose library of short, recurring French phrases (feedback,
acknowledgements, common openings) that can show up in ANY LLM reply, not
just the student's first turn -- unlike canned_responses.py, which is a
narrow, first-turn-only intent-matching dictionary and is deliberately left
untouched by this module.

match_common_phrase() only ever matches a WHOLE sentence exactly (after
normalization) against the configured list -- never a substring or isolated
word -- so a cache hit only happens when an entire LLM-generated sentence
coincides with a known phrase. That's what guarantees audio is never spliced
together from fragments.

The phrase list itself lives in backend/data/common_phrases.json -- editing
that file is the entire "expandable / configurable" mechanism; no code
change is needed to add or remove a phrase.
"""
import json
import logging
import re
import unicodedata
from typing import Optional, Tuple

from .. import config

logger = logging.getLogger(__name__)

_phrases_cache: Optional[dict] = None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _slugify(text: str) -> str:
    slug = _normalize(text).replace(" ", "_")
    return f"common_{slug}" if slug else "common_phrase"


def _load_phrases() -> dict:
    """Returns {normalized_text: canonical_text}, loaded once and cached.
    A missing or malformed file degrades to an empty dictionary (no common
    phrases matched) rather than crashing conversation turns."""
    global _phrases_cache
    if _phrases_cache is not None:
        return _phrases_cache

    _phrases_cache = {}
    path = config.COMMON_PHRASES_PATH
    if not path.exists():
        logger.warning("Common phrases file not found: %s", path)
        return _phrases_cache

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for phrase in raw:
            normalized = _normalize(phrase)
            if normalized:
                _phrases_cache[normalized] = phrase
    except Exception:
        logger.warning("Could not parse common phrases file: %s", path, exc_info=True)
        _phrases_cache = {}

    return _phrases_cache


def match_common_phrase(sentence: str) -> Optional[Tuple[str, str]]:
    """Exact normalized match only. Returns (cache_key, canonical_text) or
    None. The caller keeps showing the LLM's original sentence text to the
    student -- only the AUDIO source changes on a hit."""
    normalized = _normalize(sentence)
    if not normalized:
        return None
    phrases = _load_phrases()
    canonical = phrases.get(normalized)
    if canonical is None:
        return None
    return _slugify(canonical), canonical


def all_entries_for_precompute() -> list:
    """Used by scripts/precompute_canned_audio.py to warm every teacher's
    cache ahead of time. Teacher-name-independent (unlike
    canned_responses.all_entries_for_precompute), since these are generic
    feedback/opener phrases with no {name} placeholder."""
    return [{"key": _slugify(text), "text": text} for text in _load_phrases().values()]
