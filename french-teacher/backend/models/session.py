"""
In-memory conversation session store.

Kept intentionally simple (a dict) for the MVP. Swap SessionStore's internals
for a SQLite/Postgres-backed implementation later without touching call sites --
everything else only calls get_or_create()/get() on the store returned by
get_session_store().
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class Session:
    id: str
    level: str
    teacher_voice: str
    history: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # Set once the student names one of their teacher's configured topics
    # (see services/topics.py) -- None means still in general/open
    # conversation. Stores the topic id, not the whole dict, so Session stays
    # a small, easily-serializable record; callers resolve the full topic via
    # topics.load_teacher_topics(session.teacher_voice).
    topic: Optional[str] = None
    # Index into that topic's prepared_lines.questions -- how many prepared
    # questions have been asked so far in the current topic quiz (see
    # services/topics.py advance_topic_quiz). Meaningless while topic is None.
    topic_question_index: int = 0

    def add_turn(self, role: str, text: str) -> None:
        self.history.append(Turn(role=role, text=text))
        self.last_active = time.time()


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: Optional[str], level: str, teacher_voice: str) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_id = session_id or str(uuid.uuid4())
        session = Session(id=new_id, level=level, teacher_voice=teacher_voice)
        self._sessions[new_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def snapshot(self) -> list:
        """A point-in-time copy, safe to iterate while other threads mutate
        the live dict (e.g. the periodic cleanup sweep running alongside
        in-flight request threads)."""
        return list(self._sessions.values())


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
