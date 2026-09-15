"""
POST /api/tts/generate: direct text -> cloned-voice audio, for the "Générer
des phrases" mode (see backend/services/sentence_generation.py). Independent
of the conversation pipeline -- no session, no STT, no LLM call.

GET /api/audio/generated/{filename}: serves a generated file for preview
playback and download. Separate route/directory from the conversation
pipeline's /api/audio/response/{filename} (see config.GENERATED_AUDIO_DIR)
since these files are meant to be kept, not swept after 30 minutes.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse

from .. import config
from ..services.sentence_generation import SentenceGenerationError, generate_sentence_audio

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/tts/generate")
async def tts_generate(text: str = Form(...), teacher: str = Form(...)):
    try:
        result = generate_sentence_audio(text, teacher)
    except SentenceGenerationError as e:
        logger.warning("Sentence generation failed: %s", e)
        raise HTTPException(e.status_code, str(e)) from e
    return result


@router.get("/api/audio/generated/{filename}")
async def get_generated_audio(filename: str):
    # Filenames are server-generated (uuid-based); still guard against path traversal.
    safe_name = Path(filename).name
    file_path = config.GENERATED_AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "Audio not found.")
    return FileResponse(str(file_path), media_type="audio/wav")
