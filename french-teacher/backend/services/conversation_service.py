"""
Orchestrates one conversation turn: STT -> LLM -> TTS, with per-stage timing
logged as called for in the project brief (STT/LLM/TTS/TOTAL).
"""
import json
import logging
import queue
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Iterator, Optional

import soundfile as sf

from .. import config
from ..models.session import Session, get_session_store
from . import common_phrases
from .canned_responses import find_canned_reply
from .chatterbox_service import ChatterboxError, get_chatterbox_service
from .llm_service import LLMError, get_llm_provider
from .prompts import build_system_prompt
from .stt_service import STTError, get_stt_service
from .topics import advance_topic_quiz, find_topic_pick, start_topic_quiz, topic_menu_text
from .video_library import VideoRef, find_relevant_video

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Two kinds of soft break, tried together and picked by position (see
# _next_chunk_split): right after a comma/semicolon/colon, or right before one
# of a small set of common French coordinating conjunctions -- deliberately
# limited to these few (not "que"/"qui", which are too common inside
# subordinate clauses to split on safely) so a break never lands somewhere
# that reads as an awkward mid-phrase cut.
_PHRASE_BREAK_RE = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:et|mais|ou|donc|car|puis)\s)")

# Sentinel pushed onto the producer/consumer queue in process_message_stream
# to mark a clean end of the LLM stream (as opposed to an LLMError instance,
# which marks a failed one). A private object rather than None so it can never
# collide with a real (falsy but valid) queue item.
_STREAM_DONE = object()


def _split_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def _next_chunk_split(buffer: str, min_chars: int, max_chars: int):
    """Finds the next natural place to cut `buffer` into a speech chunk, or
    None if nothing suitable is available yet.

    A completed sentence (.!?) always wins, regardless of length -- this is
    the same boundary _split_sentences uses, so short/normal replies chunk
    exactly as before. Only once the buffer has grown past `max_chars`
    without a sentence ending do we look for a comma/semicolon/colon instead,
    picking the LATEST one that still leaves at least `min_chars` in the
    chunk -- this is what stops one long sentence from single-handedly
    delaying Time To First Audio, without ever emitting an unnaturally short
    fragment (see config.MIN_TTS_CHUNK_CHARS / MAX_TTS_CHUNK_CHARS)."""
    m = _SENTENCE_SPLIT_RE.search(buffer)
    if m:
        return buffer[: m.start()], buffer[m.end() :]

    if len(buffer) >= max_chars:
        best = None
        for m in _PHRASE_BREAK_RE.finditer(buffer):
            if m.start() < min_chars:
                continue
            if m.start() > max_chars:
                break
            best = m
        if best:
            return buffer[: best.start()], buffer[best.end() :]

    return None


def _stream_chunks(chunks: Iterator[str], min_chars: int = None, max_chars: int = None) -> Iterator[str]:
    """Incrementally detects natural speech chunks in a growing buffer of LLM
    text deltas (see _next_chunk_split for the boundary rules). Re-scans the
    whole accumulated buffer on every delta so a boundary split across two
    deltas (e.g. a '.' in one chunk, the following space in the next) is
    still detected correctly. Flushes any trailing partial buffer once the
    stream ends."""
    min_chars = config.MIN_TTS_CHUNK_CHARS if min_chars is None else min_chars
    max_chars = config.MAX_TTS_CHUNK_CHARS if max_chars is None else max_chars
    buffer = ""
    for delta in chunks:
        buffer += delta
        while True:
            result = _next_chunk_split(buffer, min_chars, max_chars)
            if result is None:
                break
            chunk, buffer = result
            if chunk.strip():
                yield chunk.strip()
    if buffer.strip():
        yield buffer.strip()


def _mentions_video(text: str) -> bool:
    """True if the student's message plausibly asks for a video (deliberately
    broad -- just the word "video"/"vidéo" appearing anywhere, accent- and
    case-insensitive -- rather than trying to enumerate every possible
    phrasing like "tu as une vidéo ?" / "recommande-moi une vidéo" / etc.)."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return "video" in stripped


def _maybe_find_video_ref(session: Session, transcript: str) -> Optional[VideoRef]:
    """Only looks up a video when the student actually asked for one (a blind
    per-turn search would risk the LLM being nudged to mention videos when
    nothing was asked). Searches recent conversation context, not just the
    trigger message alone, since "tu as une vidéo sur ce sujet ?" carries no
    topic keywords by itself -- the topic is whatever was just discussed."""
    if not _mentions_video(transcript):
        return None
    context_text = " ".join(t.text for t in session.history[-6:])
    return find_relevant_video(context_text)


def _video_prompt_addendum(video_ref: Optional[VideoRef]) -> str:
    if not video_ref:
        return ""
    return (
        f"\n\nL'étudiant(e) demande une vidéo. Une vidéo pertinente est disponible, intitulée "
        f"« {video_ref.title} ». Mentionne ce titre naturellement dans ta réponse pour la lui "
        "recommander. Ne dis pas l'URL à voix haute -- le lien sera affiché séparément à l'écran."
    )


def _cached_audio_path(teacher_voice: str, cache_key: str) -> Path:
    return config.VOICES_DIR / teacher_voice / "canned_audio" / f"{cache_key}.wav"


def _synthesize(chatterbox, teacher_voice: str, reference_path: Path, text: str, cache_key: Optional[str] = None):
    """Generates (or reuses a cached) clip for `text` in the teacher's voice.

    Only a handful of lines are ever fully predictable per session (the fixed
    greeting, and a few canned first-turn replies -- see canned_responses.py);
    those are the only callers that pass `cache_key`. Everything the LLM writes
    is unique per turn and always goes straight to Chatterbox, uncached.
    """
    if cache_key:
        cache_path = _cached_audio_path(teacher_voice, cache_key)
        if cache_path.exists():
            wav_np, sr = sf.read(str(cache_path), dtype="int16")
            return wav_np, sr

    wav_np, sr = chatterbox.generate(text, str(reference_path), language=config.CHATTERBOX_LANGUAGE)

    if cache_key:
        cache_path = _cached_audio_path(teacher_voice, cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(cache_path), wav_np, sr, subtype="PCM_16")

    return wav_np, sr


class ConversationError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _teacher_display_name(teacher_voice: str) -> str:
    meta_path = config.VOICES_DIR / teacher_voice / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")).get("display_name", teacher_voice)
        except Exception:
            logger.warning("Could not read meta.json for %s", teacher_voice, exc_info=True)
    return teacher_voice


def _reference_audio_path(teacher_voice: str) -> Path:
    return config.VOICES_DIR / teacher_voice / "reference.wav"


def _resolve_reply_shortcut(session: Session, transcript: str, teacher_name: str) -> Optional[dict]:
    """Returns a canned-reply-shaped dict ({"key", "text"}) if this turn can
    be answered without an LLM call, or None if the LLM should handle it as
    normal. Three cases, checked in order:

    1. session.topic is already set -- an in-progress topic quiz (see
       services/topics.py) always continues with the next prepared question,
       never the LLM, regardless of turn number.
    2. The student's very first turn matches a small-talk opener
       (canned_responses.py) -- unrelated to topics, this already existed.
    3. The transcript names one of this teacher's topics (find_topic_pick) --
       checked on ANY turn where a topic isn't set yet, not just the first,
       since a student might chat a little before picking one.
    """
    if session.topic:
        return advance_topic_quiz(session, transcript)

    is_first_user_turn = sum(1 for t in session.history if t.role == "user") == 1
    if is_first_user_turn:
        canned = find_canned_reply(transcript, teacher_name, topic_menu_text(session.teacher_voice))
        if canned:
            return canned

    picked = find_topic_pick(transcript, session.teacher_voice)
    if picked:
        return start_topic_quiz(session, picked)

    return None


def _serialize_videos(videos: Optional[list]) -> Optional[list]:
    if not videos:
        return None
    return [{"title": v.title, "url": v.url, "thumbnail_url": v.thumbnail_url} for v in videos]


def _delete_session_audio(session_id: str) -> int:
    """Deletes every generated audio file for one session. Only ever touches
    config.OUTPUT_DIR (output/conversation_audio/) -- never config.VOICES_DIR
    -- so reference.wav/raw.wav/meta.json/canned_audio are untouched by
    construction, not by a special-case check."""
    removed = 0
    for path in config.OUTPUT_DIR.glob(f"{session_id}_*.wav"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def end_conversation(session_id: str) -> None:
    """Explicit-end path: deletes the session's audio and removes it from the
    store. Idempotent -- a repeat call (or a call for an id that never
    existed) is a harmless no-op, since the frontend's pagehide beacon can't
    react to an error response anyway."""
    _delete_session_audio(session_id)
    get_session_store().delete(session_id)


def cleanup_expired_sessions() -> int:
    """Fallback for sessions the pagehide beacon never reaches (browser
    crash, force-quit, etc). Two passes, both scoped to config.OUTPUT_DIR:

    1. Any session idle longer than SESSION_TTL_SECONDS is deleted the same
       way end_conversation() would.
    2. A second, session-independent sweep deletes any leftover .wav older
       than SESSION_TTL_SECONDS by mtime. This closes a real race: if a tab
       closes mid-turn, the in-flight request keeps running to completion in
       its own thread (FastAPI/Starlette don't cancel a blocking sync
       generator just because the client disconnected) and can write a new
       file *after* end_conversation already ran for that session -- orphaning
       it forever since the session entry is already gone. The mtime sweep is
       self-healing against that without needing to cancel in-flight requests.

    Returns the number of sessions removed (for logging)."""
    store = get_session_store()
    now = time.time()
    removed_sessions = 0

    for session in store.snapshot():
        if now - session.last_active > config.SESSION_TTL_SECONDS:
            _delete_session_audio(session.id)
            store.delete(session.id)
            removed_sessions += 1

    if config.OUTPUT_DIR.exists():
        for path in config.OUTPUT_DIR.glob("*.wav"):
            try:
                if now - path.stat().st_mtime > config.SESSION_TTL_SECONDS:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    if removed_sessions:
        logger.info("Cleanup swept %d expired session(s).", removed_sessions)
    return removed_sessions


def build_greeting_text(teacher_name: str, topic_menu: str) -> str:
    """Single source of truth for the fixed opening line -- also used by
    scripts/precompute_canned_audio.py, so both places can never drift apart
    the way canned_responses.py's replies once did against their own cache.

    topic_menu is this teacher's own spoken topic list (see topics.py) --
    every teacher used to share one hardcoded generic menu regardless of who
    they were; callers now resolve it per teacher_voice via topic_menu_text()."""
    return (
        f"Bonjour, c'est {teacher_name}. Ceci est une copie générée par intelligence artificielle. "
        f"On commence ? Choisis un thème : {topic_menu}."
    )


def start_conversation(level: str, teacher_voice: str) -> dict:
    """Greets the student first, in the teacher's own voice, disclosing that it's an
    AI clone -- required by this project's voice-cloning consent rules (see README)."""
    store = get_session_store()
    session = store.get_or_create(None, level=level, teacher_voice=teacher_voice)

    teacher_name = _teacher_display_name(session.teacher_voice)
    greeting_text = build_greeting_text(teacher_name, topic_menu_text(session.teacher_voice))
    session.add_turn("assistant", greeting_text)

    t0 = time.perf_counter()
    chatterbox = get_chatterbox_service()
    reference_path = _reference_audio_path(session.teacher_voice)
    try:
        wav_np, sr = _synthesize(chatterbox, session.teacher_voice, reference_path, greeting_text, cache_key="greeting")
    except ChatterboxError as e:
        raise ConversationError(f"La synthèse vocale a échoué : {e}", 502) from e
    t_tts = time.perf_counter() - t0

    audio_id = f"{session.id}_{uuid.uuid4().hex[:8]}.wav"
    audio_path_out = config.OUTPUT_DIR / audio_id
    sf.write(str(audio_path_out), wav_np, sr, subtype="PCM_16")

    logger.info("Greeting TTS: %.2fs", t_tts)

    return {
        "session_id": session.id,
        "response_text": greeting_text,
        "audio_url": f"/api/audio/response/{audio_id}",
        "timing": {"tts": round(t_tts, 2), "total": round(t_tts, 2)},
    }


def process_message(
    session_id: Optional[str],
    level: str,
    teacher_voice: str,
    audio_path: Optional[str] = None,
    text_override: Optional[str] = None,
) -> dict:
    """Either audio_path (recorded speech) or text_override (typed fallback) must be given."""
    store = get_session_store()
    session = store.get_or_create(session_id, level=level, teacher_voice=teacher_voice)

    t0 = time.perf_counter()
    if text_override is not None:
        transcript = text_override.strip()
        t_stt = 0.0
    else:
        stt = get_stt_service()
        try:
            transcript = stt.transcribe(audio_path, language="fr")
        except STTError as e:
            raise ConversationError(f"Speech recognition failed: {e}", 502) from e
        t_stt = time.perf_counter() - t0

    if not transcript:
        raise ConversationError("Je n'ai rien entendu -- peux-tu réessayer ?", 422)

    session.add_turn("user", transcript)

    t1 = time.perf_counter()
    teacher_name = _teacher_display_name(session.teacher_voice)
    canned = _resolve_reply_shortcut(session, transcript, teacher_name)
    video_ref = _maybe_find_video_ref(session, transcript)
    if canned:
        reply_text = canned["text"]
        t_llm = 0.0
    else:
        system_prompt = build_system_prompt(session.level, teacher_name) + _video_prompt_addendum(video_ref)
        history = [{"role": t.role, "text": t.text} for t in session.history]
        try:
            llm = get_llm_provider()
            reply_text = llm.generate_reply(system_prompt, history)
        except LLMError as e:
            raise ConversationError(f"Le cerveau (LLM) du professeur n'a pas répondu : {e}", 502) from e
        t_llm = time.perf_counter() - t1
    session.add_turn("assistant", reply_text)

    t2 = time.perf_counter()
    chatterbox = get_chatterbox_service()
    reference_path = _reference_audio_path(session.teacher_voice)
    try:
        wav_np, sr = _synthesize(
            chatterbox, session.teacher_voice, reference_path, reply_text,
            cache_key=canned["key"] if canned else None,
        )
    except ChatterboxError as e:
        raise ConversationError(f"La synthèse vocale a échoué : {e}", 502) from e
    t_tts = time.perf_counter() - t2

    audio_id = f"{session.id}_{uuid.uuid4().hex[:8]}.wav"
    audio_path_out = config.OUTPUT_DIR / audio_id
    sf.write(str(audio_path_out), wav_np, sr, subtype="PCM_16")

    t_total = time.perf_counter() - t0
    logger.info("STT: %.2fs | LLM: %.2fs | TTS: %.2fs | TOTAL: %.2fs", t_stt, t_llm, t_tts, t_total)

    return {
        "session_id": session.id,
        "transcript": transcript,
        "response_text": reply_text,
        "audio_url": f"/api/audio/response/{audio_id}",
        "video": {"title": video_ref.title, "url": video_ref.url, "thumbnail_url": video_ref.thumbnail_url} if video_ref else None,
        "topic_videos": _serialize_videos(canned.get("videos") if canned else None),
        "timing": {
            "stt": round(t_stt, 2),
            "llm": round(t_llm, 2),
            "tts": round(t_tts, 2),
            "total": round(t_total, 2),
        },
    }


def process_message_stream(
    session_id: Optional[str],
    level: str,
    teacher_voice: str,
    audio_path: Optional[str] = None,
    text_override: Optional[str] = None,
) -> Iterator[dict]:
    """Generator version of process_message: yields one event dict as soon as each
    piece is ready (transcript, then each synthesized sentence, then done/error).

    TTS is by far the slowest stage (typically 5-6x realtime -- see the STT/LLM/TTS
    timing logged per turn), and it dominates total turn time far more than the LLM
    does. On a non-canned turn, the LLM's reply is consumed as a stream on a
    background thread (_produce below) that detects natural speech chunks
    incrementally (_stream_chunks -- sentence boundaries, or a comma/phrase
    boundary if one sentence runs long) and pushes each one onto a queue as soon
    as it's ready. THIS thread -- the only thread that ever calls Chatterbox, by
    construction -- consumes that queue in order and synthesizes each chunk. The
    result is that while chunk N is being synthesized, the LLM keeps streaming
    chunk N+1's text in the background instead of both sides sitting idle waiting
    on each other; ordering falls out for free from the queue being FIFO with a
    single consumer, and "never more than one Chatterbox job at a time" falls out
    for free from that same single consumer. Each completed chunk is also checked
    against the common-phrase cache (common_phrases.py) so frequent short lines
    reuse pre-generated audio instead of calling Chatterbox again. Never raises
    ConversationError once at least one event has been yielded (the HTTP response
    is already committed by then); failures become an "error" event instead so
    the stream always ends cleanly.

    Time To First Audio (TTFA) -- the time between the student finishing speaking
    and the first synthesized chunk being ready -- is the headline latency
    metric; see the "ttfa" field in the final "done" event's timing.
    """
    store = get_session_store()
    session = store.get_or_create(session_id, level=level, teacher_voice=teacher_voice)

    t0 = time.perf_counter()
    if text_override is not None:
        transcript = text_override.strip()
        t_stt = 0.0
    else:
        stt = get_stt_service()
        try:
            transcript = stt.transcribe(audio_path, language="fr")
        except STTError as e:
            yield {"type": "error", "message": f"Speech recognition failed: {e}"}
            return
        t_stt = time.perf_counter() - t0

    if not transcript:
        yield {"type": "error", "message": "Je n'ai rien entendu -- peux-tu réessayer ?"}
        return

    session.add_turn("user", transcript)
    yield {"type": "transcript", "session_id": session.id, "transcript": transcript}

    teacher_name = _teacher_display_name(session.teacher_voice)
    canned = _resolve_reply_shortcut(session, transcript, teacher_name)
    video_ref = _maybe_find_video_ref(session, transcript)

    chatterbox = get_chatterbox_service()
    reference_path = _reference_audio_path(session.teacher_voice)

    t_llm = 0.0
    t_llm_first_chunk = None
    t_tts = 0.0
    t_tts_first_chunk = None
    ttfa = None
    cache_hits = 0
    cache_misses = 0
    sentence_texts = []

    def _emit(sentence_text: str, cache_key: Optional[str], index: int, count: Optional[int]) -> dict:
        nonlocal t_tts, t_tts_first_chunk, ttfa, cache_hits, cache_misses
        is_cache_hit = bool(cache_key) and _cached_audio_path(session.teacher_voice, cache_key).exists()

        t2 = time.perf_counter()
        wav_np, sr = _synthesize(chatterbox, session.teacher_voice, reference_path, sentence_text, cache_key=cache_key)
        chunk_tts_time = time.perf_counter() - t2
        t_tts += chunk_tts_time
        if t_tts_first_chunk is None:
            t_tts_first_chunk = chunk_tts_time

        if is_cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        audio_id = f"{session.id}_{uuid.uuid4().hex[:8]}.wav"
        sf.write(str(config.OUTPUT_DIR / audio_id), wav_np, sr, subtype="PCM_16")

        if ttfa is None:
            ttfa = time.perf_counter() - t0

        return {
            "type": "sentence",
            "index": index,
            "count": count,
            "text": sentence_text,
            "audio_url": f"/api/audio/response/{audio_id}",
            "cache_hit": is_cache_hit,
        }

    def _timed_chunks(raw_iter):
        """Wraps the raw LLM stream to measure PURE LLM wait time -- only time
        actually spent blocked in next() -- so interleaved TTS work between
        sentences below doesn't inflate the LLM timing metric."""
        nonlocal t_llm, t_llm_first_chunk
        it = iter(raw_iter)
        first = True
        while True:
            start = time.perf_counter()
            try:
                chunk = next(it)
            except StopIteration:
                return
            t_llm += time.perf_counter() - start
            if first:
                t_llm_first_chunk = t_llm
                first = False
            if chunk:
                yield chunk

    try:
        if canned:
            sentences = _split_sentences(canned["text"])
            for i, sentence in enumerate(sentences):
                sentence_texts.append(sentence)
                yield _emit(sentence, f"{canned['key']}_{i}", i, len(sentences))
        else:
            system_prompt = build_system_prompt(session.level, teacher_name) + _video_prompt_addendum(video_ref)
            history = [{"role": t.role, "text": t.text} for t in session.history]
            llm = get_llm_provider()

            # Background producer: pulls the LLM stream and detects chunk
            # boundaries, completely decoupled from Chatterbox -- this is what
            # lets LLM generation (network-bound) and TTS synthesis (GPU-bound)
            # overlap instead of strictly alternating. See the docstring above.
            chunk_queue: "queue.Queue" = queue.Queue()

            def _produce():
                nonlocal t_llm
                try:
                    # generate_reply_stream() itself can block for a while before
                    # returning an iterator at all (SDK request setup, TLS, time
                    # to first response headers) -- time that entirely without it,
                    # "llm"/"llm_first_chunk" would under-report real latency.
                    t_setup_start = time.perf_counter()
                    raw_stream = llm.generate_reply_stream(system_prompt, history)
                    t_llm += time.perf_counter() - t_setup_start
                    for chunk_text in _stream_chunks(_timed_chunks(raw_stream)):
                        chunk_queue.put(chunk_text)
                except LLMError as e:
                    chunk_queue.put(e)
                    return
                except Exception as e:  # defensive: never hang the consumer on a surprise error
                    chunk_queue.put(LLMError(f"LLM streaming failed: {e}"))
                    return
                chunk_queue.put(_STREAM_DONE)

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()

            i = 0
            while True:
                item = chunk_queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, LLMError):
                    raise item
                sentence_texts.append(item)
                match = common_phrases.match_common_phrase(item)
                cache_key = match[0] if match else None
                yield _emit(item, cache_key, i, None)
                i += 1
            producer.join()
    except LLMError as e:
        yield {"type": "error", "message": f"Le cerveau (LLM) du professeur n'a pas répondu : {e}"}
        return
    except ChatterboxError as e:
        yield {"type": "error", "message": f"La synthèse vocale a échoué : {e}"}
        return

    if not sentence_texts:
        yield {"type": "error", "message": "Le professeur n'a rien répondu -- peux-tu réessayer ?"}
        return

    reply_text = " ".join(sentence_texts)
    session.add_turn("assistant", reply_text)

    t_total = time.perf_counter() - t0
    logger.info(
        "STT: %.2fs | LLM: %.2fs (first chunk %.2fs) | TTS: %.2fs (first chunk %.2fs) | "
        "TTFA: %.2fs | chunks: %d (cache hit=%d miss=%d) | TOTAL: %.2fs",
        t_stt, t_llm, t_llm_first_chunk or 0.0, t_tts, t_tts_first_chunk or 0.0,
        ttfa or 0.0, len(sentence_texts), cache_hits, cache_misses, t_total,
    )
    yield {
        "type": "done",
        "response_text": reply_text,
        "video": {"title": video_ref.title, "url": video_ref.url, "thumbnail_url": video_ref.thumbnail_url} if video_ref else None,
        "topic_videos": _serialize_videos(canned.get("videos") if canned else None),
        "timing": {
            "stt": round(t_stt, 2),
            "llm": round(t_llm, 2),
            "llm_first_chunk": round(t_llm_first_chunk, 2) if t_llm_first_chunk is not None else None,
            "tts": round(t_tts, 2),
            "tts_first_chunk": round(t_tts_first_chunk, 2) if t_tts_first_chunk is not None else None,
            "chunks": len(sentence_texts),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "ttfa": round(ttfa, 2) if ttfa is not None else None,
            "total": round(t_total, 2),
        },
    }
