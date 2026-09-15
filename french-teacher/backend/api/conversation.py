"""
POST /api/conversation/message: the main conversation turn endpoint.
GET  /api/audio/response/{filename}: serves generated teacher audio.
GET  /api/teachers: lists configured teacher voices for the frontend's selector.

Internal model/service details (Chatterbox, Whisper, LLM provider) are never
exposed to the frontend -- only transcript/response_text/audio_url/timing.
"""
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .. import config
from ..services.conversation_service import (
    ConversationError,
    end_conversation,
    process_message,
    process_message_stream,
    start_conversation,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_level_and_teacher(level: str, teacher: str) -> None:
    if level not in config.VALID_LEVELS:
        raise HTTPException(400, f"Invalid level '{level}'. Must be one of {config.VALID_LEVELS}.")

    teacher_dir = config.VOICES_DIR / teacher
    if not (teacher_dir / "reference.wav").exists():
        raise HTTPException(400, f"Unknown or unconfigured teacher voice '{teacher}'.")


@router.post("/api/conversation/start")
async def conversation_start(
    level: str = Form(config.DEFAULT_STUDENT_LEVEL),
    teacher: str = Form(config.ACTIVE_TEACHER_VOICE),
):
    """Greets the student first, before any student turn -- see start_conversation()."""
    _validate_level_and_teacher(level, teacher)
    try:
        result = start_conversation(level, teacher)
    except ConversationError as e:
        logger.warning("Conversation start failed: %s", e)
        raise HTTPException(e.status_code, str(e)) from e
    return result


@router.post("/api/conversation/message")
async def conversation_message(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    level: str = Form(config.DEFAULT_STUDENT_LEVEL),
    teacher: str = Form(config.ACTIVE_TEACHER_VOICE),
):
    _validate_level_and_teacher(level, teacher)

    if audio is None and not text:
        raise HTTPException(422, "Provide either a recorded 'audio' file or 'text'.")

    tmp_path = None
    try:
        if text:
            result = process_message(session_id, level, teacher, text_override=text)
        else:
            raw = await audio.read()
            if not raw:
                raise HTTPException(422, "Empty audio recording received.")
            if len(raw) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Recording is too long. Please keep messages under a minute.")

            suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            result = process_message(session_id, level, teacher, audio_path=tmp_path)
    except ConversationError as e:
        logger.warning("Conversation turn failed: %s", e)
        raise HTTPException(e.status_code, str(e)) from e
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return result


@router.post("/api/conversation/message/stream")
async def conversation_message_stream(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    level: str = Form(config.DEFAULT_STUDENT_LEVEL),
    teacher: str = Form(config.ACTIVE_TEACHER_VOICE),
):
    """Same request shape as /api/conversation/message, but streams newline-delimited
    JSON events (transcript, then one per synthesized sentence, then done/error) as
    soon as each is ready instead of waiting for the whole reply -- see
    process_message_stream() for why this matters (TTS dominates turn latency)."""
    _validate_level_and_teacher(level, teacher)

    if audio is None and not text:
        raise HTTPException(422, "Provide either a recorded 'audio' file or 'text'.")

    tmp_path = None
    if not text:
        raw = await audio.read()
        if not raw:
            raise HTTPException(422, "Empty audio recording received.")
        if len(raw) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Recording is too long. Please keep messages under a minute.")

        suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

    def event_stream():
        try:
            if text:
                gen = process_message_stream(session_id, level, teacher, text_override=text)
            else:
                gen = process_message_stream(session_id, level, teacher, audio_path=tmp_path)
            for event in gen:
                yield json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/api/conversation/end")
async def conversation_end(session_id: str = Form(...)):
    """Explicit-end: deletes the session's audio and its SessionStore entry.
    Called by the frontend's pagehide beacon on tab close/navigate away.
    Always succeeds -- idempotent for an already-gone or unknown session_id,
    since navigator.sendBeacon can't react to an error response anyway."""
    end_conversation(session_id)
    return {"ok": True}


@router.get("/api/audio/response/{filename}")
async def get_response_audio(filename: str):
    # Filenames are server-generated (uuid-based); still guard against path traversal.
    safe_name = Path(filename).name
    file_path = config.OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "Audio not found.")
    return FileResponse(str(file_path), media_type="audio/wav")


@router.get("/api/teachers")
async def list_teachers():
    teachers = []
    if config.VOICES_DIR.exists():
        for d in sorted(config.VOICES_DIR.iterdir()):
            if d.is_dir() and (d / "reference.wav").exists():
                display_name = d.name
                meta_path = d / "meta.json"
                if meta_path.exists():
                    try:
                        display_name = json.loads(meta_path.read_text(encoding="utf-8")).get(
                            "display_name", d.name
                        )
                    except Exception:
                        logger.warning("Could not read meta.json for %s", d.name, exc_info=True)
                teachers.append({"id": d.name, "display_name": display_name})
    return {"teachers": teachers, "active_default": config.ACTIVE_TEACHER_VOICE}
