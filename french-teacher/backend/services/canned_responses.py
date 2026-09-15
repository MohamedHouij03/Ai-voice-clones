"""
A small, curated dictionary of ultra-predictable opening exchanges.

The very first thing a student says after the teacher's greeting is highly
predictable ("Bonjour", "Ça va, et toi ?", etc.) -- there's no need to pay for
an LLM call and a fresh 10-40s Chatterbox generation for lines this
foreseeable. find_canned_reply() is only ever checked against the student's
very first turn in a session (see conversation_service.py); anything not
confidently matched here falls straight through to the real LLM, unchanged.

The teacher's greeting (build_greeting_text() in conversation_service.py)
already offers a topic menu once ("Choisis un thème : la nourriture, le
cinéma, les voyages, ou le sport."). These replies fire when the student's
first turn is small talk instead of picking one -- they must NOT ask an
open-ended "what do you want to talk about?" (too hard a question to
generate an answer to from scratch, especially at low levels) and must NOT
repeat the exact same offer verbatim (that produced an actual "ready? / yes
/ ready? / yes"-style loop in testing before): acknowledge briefly, then
re-offer the same concrete menu.
"""
import re
import unicodedata
from typing import Optional, TypedDict


class CannedReply(TypedDict):
    key: str
    text: str


_ENTRIES = [
    {
        "key": "ca_va",
        "triggers": [
            "ca va bien et toi",
            "ca va tres bien et toi",
            "ca va et toi",
            "ca va bien",
            "ca va tres bien",
            "ca va",
        ],
        "reply": "Ça va très bien, merci ! Alors, choisis un thème : {menu}.",
    },
    {
        "key": "comment_vas_tu",
        "triggers": ["comment vas tu", "comment allez vous", "comment ca va"],
        "reply": "Je vais très bien, merci de demander ! Et toi, choisis un thème : {menu}.",
    },
    {
        "key": "ton_nom",
        "triggers": [
            "comment tu t appelles",
            "comment vous appelez vous",
            "quel est ton nom",
            "quel est votre nom",
            "tu t appelles comment",
        ],
        # Answers the teacher's OWN (persona) name, which is fine to state --
        # deliberately does NOT ask the student's name back; see the privacy
        # rule in prompts.py.
        "reply": "Je m'appelle {name}, ton professeur de français. Alors, choisis un thème : {menu}.",
    },
    {
        "key": "salut_only",
        "triggers": ["bonjour", "salut", "coucou", "bonsoir"],
        "reply": "Bonjour à toi aussi ! Alors, choisis un thème : {menu}.",
    },
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_canned_reply(transcript: str, teacher_name: str, topic_menu: str) -> Optional[CannedReply]:
    normalized = _normalize(transcript)
    if not normalized:
        return None
    for entry in _ENTRIES:
        if any(normalized == trig or normalized.startswith(trig) for trig in entry["triggers"]):
            return {"key": entry["key"], "text": entry["reply"].format(name=teacher_name, menu=topic_menu)}
    return None


def all_entries_for_precompute(teacher_name: str, topic_menu: str) -> list:
    """Used by scripts/precompute_canned_audio.py to warm the cache for every
    known teacher ahead of time, so even the *first* real session is fast."""
    return [
        {"key": e["key"], "text": e["reply"].format(name=teacher_name, menu=topic_menu)}
        for e in _ENTRIES
    ]
