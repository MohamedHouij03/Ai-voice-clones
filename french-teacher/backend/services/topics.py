"""
Per-teacher conversation topics (see voices/<teacher>/topics.json), used to
build a teacher-specific spoken topic menu instead of every teacher sharing
one generic list. A teacher with no topics.json configured yet falls back to
the original generic menu, not an error -- see topic_menu_text().

Also implements the topic quiz engine (find_topic_pick / start_topic_quiz /
advance_topic_quiz): once a student names one of their teacher's topics, the
conversation switches to walking through that topic's PRE-WRITTEN
question/expected_answers pairs (see each topic's prepared_lines.questions in
topics.json) instead of calling the LLM at all -- deliberately, so a topic
quiz is instant and free of live generation, unlike general conversation.
Answer checking is plain normalized string matching against expected_answers,
same spirit as canned_responses.py's trigger matching -- not an LLM judgment
call. A question with an empty expected_answers list is a preference/opinion
question (e.g. "Aimes-tu les jeux vidéo ?") -- any answer is acknowledged,
never marked right or wrong. Once a topic's questions run out, that teacher's
own video recommendations (see teacher_videos.py) are attached to the
wrap-up reply, if any are configured.

Imports only config, models, and teacher_videos (itself config-only) -- never
conversation_service.py or canned_responses.py -- so both of those can import
FROM this module without a circular import.
"""
import json
import logging
import re
import unicodedata
from typing import Optional

from .. import config
from ..models.session import Session
from .teacher_videos import teacher_video_recommendations

logger = logging.getLogger(__name__)

_GENERIC_MENU = "la nourriture, le cinéma, les voyages, ou le sport"

_CORRECT_REACTION = "Exactement !"
_INCORRECT_REACTION_TEMPLATE = "Pas tout à fait, on dit plutôt « {answer} »."
_OPEN_REACTION = "D'accord !"

# topics.json is authored offline (not edited while the server is running --
# same assumption common_phrases.py already makes for its own cache), so a
# per-process cache is safe and avoids re-reading the file every turn.
_topics_cache: dict = {}


def load_teacher_topics(teacher_voice: str) -> list:
    """Returns this teacher's configured topics (a list of dicts, see
    voices/<teacher>/topics.json), or [] if none are configured."""
    if teacher_voice in _topics_cache:
        return _topics_cache[teacher_voice]

    topics_path = config.VOICES_DIR / teacher_voice / "topics.json"
    topics = []
    if topics_path.exists():
        try:
            topics = json.loads(topics_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse topics.json for %s", teacher_voice, exc_info=True)
            topics = []
    _topics_cache[teacher_voice] = topics
    return topics


def topic_menu_text(teacher_voice: str) -> str:
    """A natural, comma-joined spoken list of this teacher's topic labels
    (lowercased to fit mid-sentence, e.g. "...choisis un thème : cinéma et
    films, films français, ou métiers de rêve et carrières."), or the
    original generic menu if this teacher has no topics configured."""
    labels = [t["label"] for t in load_teacher_topics(teacher_voice) if t.get("label")]
    if not labels:
        return _GENERIC_MENU
    lowered = [label[0].lower() + label[1:] for label in labels]
    if len(lowered) == 1:
        return lowered[0]
    return ", ".join(lowered[:-1]) + f", ou {lowered[-1]}"


def get_topic_by_id(teacher_voice: str, topic_id: str) -> Optional[dict]:
    for topic in load_teacher_topics(teacher_voice):
        if topic.get("id") == topic_id:
            return topic
    return None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_topic_pick(transcript: str, teacher_voice: str) -> Optional[dict]:
    """Matches free-form student text (e.g. "on parle de cinéma ?", "le
    cinéma") against this teacher's topic labels/keywords -- a substring
    match on normalized text, same normalize() as canned_responses.py/
    common_phrases.py use, deliberately looser than their exact-trigger
    matching since a topic pick is more free-form than a fixed opener."""
    normalized_transcript = _normalize(transcript)
    if not normalized_transcript:
        return None
    for topic in load_teacher_topics(teacher_voice):
        candidates = [topic.get("label", "")] + topic.get("keywords", [])
        for candidate in candidates:
            normalized_candidate = _normalize(candidate)
            if normalized_candidate and normalized_candidate in normalized_transcript:
                return topic
    return None


def _evaluate_answer(question: dict, transcript: str) -> tuple:
    """Returns (outcome, reaction_text). outcome is folded into the cache key
    (see advance_topic_quiz) so each distinct reaction variant caches under
    its own key -- never reuses one teacher's "correct" audio for another
    question's "correct" audio, since the surrounding sentence differs."""
    expected = question.get("expected_answers") or []
    if not expected:
        return "open", _OPEN_REACTION
    normalized_answer = _normalize(transcript)
    if any(_normalize(exp) and _normalize(exp) in normalized_answer for exp in expected):
        return "correct", _CORRECT_REACTION
    return "incorrect", _INCORRECT_REACTION_TEMPLATE.format(answer=expected[0])


def start_topic_quiz(session: Session, topic: dict) -> Optional[dict]:
    """Called the moment a topic pick is detected. Returns a canned-reply-
    shaped dict ({"key", "text"}) plugging directly into the same cache_key
    path conversation_service.py already uses for canned_responses.py
    entries -- or None if this topic has no prepared questions yet (in which
    case the caller leaves session.topic unset and falls through to the LLM
    as normal, rather than starting an empty quiz)."""
    questions = topic.get("prepared_lines", {}).get("questions", [])
    if not questions:
        return None
    session.topic = topic["id"]
    session.topic_question_index = 1
    label = topic["label"]
    intro = label[0].lower() + label[1:]
    text = f"D'accord, parlons de {intro} ! {questions[0]['question']}"
    return {"key": f"topic_{topic['id']}_q0_start", "text": text}


def advance_topic_quiz(session: Session, transcript: str) -> Optional[dict]:
    """Called on every turn while session.topic is set. Evaluates the answer
    to whichever question was just asked, then either asks the next prepared
    question or, once the topic's questions are exhausted, hands the
    conversation back to general/open mode (clearing session.topic) and
    re-offers this teacher's topic menu."""
    topic = get_topic_by_id(session.teacher_voice, session.topic or "")
    if not topic:
        session.topic = None
        session.topic_question_index = 0
        return None

    questions = topic.get("prepared_lines", {}).get("questions", [])
    answered_idx = session.topic_question_index - 1
    if 0 <= answered_idx < len(questions):
        outcome, reaction = _evaluate_answer(questions[answered_idx], transcript)
    else:
        outcome, reaction = "start", ""

    next_idx = session.topic_question_index
    if next_idx >= len(questions):
        session.topic = None
        session.topic_question_index = 0
        menu = topic_menu_text(session.teacher_voice)
        text = f"{reaction} On a fait le tour de « {topic['label']} » ! Choisis un autre thème : {menu}.".strip()
        result = {"key": f"topic_{topic['id']}_done_{outcome}", "text": text}
        videos = teacher_video_recommendations(session.teacher_voice)
        if videos:
            result["videos"] = videos
        return result

    question = questions[next_idx]
    session.topic_question_index += 1
    text = f"{reaction} {question['question']}".strip()
    return {"key": f"topic_{topic['id']}_q{next_idx}_{outcome}", "text": text}
