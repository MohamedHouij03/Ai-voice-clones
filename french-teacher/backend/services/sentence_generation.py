"""
Direct text -> cloned-voice audio generation, completely independent of the
conversation pipeline (no STT, no LLM, no session/history) -- backs the
"Générer des phrases" mode: fixing a missed or incorrect line in a recorded
video without re-recording, using the teacher's own cloned voice. See
backend/api/tts.py for the HTTP route.

Deliberately reuses chatterbox_service.generate() exactly as-is (same
function the conversation pipeline calls) rather than adding a second TTS
code path -- this module is orchestration only, so it can't drift from
however the conversation pipeline calls Chatterbox.
"""
import logging
import uuid
from pathlib import Path

import soundfile as sf

from .. import config
from .chatterbox_service import ChatterboxError, get_chatterbox_service

logger = logging.getLogger(__name__)


class SentenceGenerationError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def teacher_reference_path(teacher_voice: str) -> Path:
    return config.VOICES_DIR / teacher_voice / "reference.wav"


def generate_sentence_audio(text: str, teacher_voice: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise SentenceGenerationError("Écris une phrase à générer.", 422)

    reference_path = teacher_reference_path(teacher_voice)
    if not reference_path.exists():
        raise SentenceGenerationError(f"Professeur inconnu ou non configuré : '{teacher_voice}'.", 400)

    chatterbox = get_chatterbox_service()
    try:
        wav_np, sr = chatterbox.generate(text, str(reference_path), language=config.CHATTERBOX_LANGUAGE)
    except ChatterboxError as e:
        raise SentenceGenerationError(f"La génération audio a échoué : {e}", 502) from e

    filename = f"{uuid.uuid4().hex}.wav"
    out_path = config.GENERATED_AUDIO_DIR / filename
    sf.write(str(out_path), wav_np, sr, subtype="PCM_16")

    logger.info("Generated sentence audio for teacher=%s (%d chars) -> %s", teacher_voice, len(text), filename)

    return {"audio_url": f"/api/audio/generated/{filename}", "text": text, "teacher": teacher_voice}
