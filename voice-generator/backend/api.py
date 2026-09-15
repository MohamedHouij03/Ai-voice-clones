"""
POST /api/generate: synthesizes a full script in a chosen cloned voice.
GET  /api/voices: lists configured voices for the frontend's selector.
GET  /api/audio/{filename}: serves a generated clip.

No STT, no LLM -- the editor supplies the exact script text they want spoken.
"""
import json
import logging
import re
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse

from . import config
from .chatterbox_service import ChatterboxError, get_chatterbox_service

logger = logging.getLogger(__name__)
router = APIRouter()

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SILENCE_GAP_SECONDS = 0.35  # brief pause between sentences when a script is split and stitched back together


def _split_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def _voice_reference_path(voice_id: str) -> Path:
    return config.VOICES_DIR / voice_id / "reference.wav"


def _voice_display_name(voice_id: str) -> str:
    meta_path = config.VOICES_DIR / voice_id / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")).get("display_name", voice_id)
        except Exception:
            logger.warning("Could not read meta.json for %s", voice_id, exc_info=True)
    return voice_id


def _validate_voice(voice_id: str) -> None:
    if not (config.VOICES_DIR / voice_id / "reference.wav").exists():
        raise HTTPException(400, f"Unknown or unconfigured voice '{voice_id}'.")


@router.get("/api/voices")
async def list_voices():
    voices = []
    if config.VOICES_DIR.exists():
        for d in sorted(config.VOICES_DIR.iterdir()):
            if d.is_dir() and (d / "reference.wav").exists():
                voices.append({"id": d.name, "display_name": _voice_display_name(d.name)})
    return {"voices": voices}


@router.post("/api/generate")
async def generate(
    voice: str = Form(...),
    text: str = Form(...),
    language: str = Form(config.CHATTERBOX_LANGUAGE),
    speed: float = Form(1.0),
    exaggeration: float = Form(config.CHATTERBOX_EXAGGERATION),
):
    _validate_voice(voice)
    text = text.strip()
    if not text:
        raise HTTPException(422, "Script text is empty.")
    if len(text) > config.MAX_SCRIPT_CHARS:
        raise HTTPException(413, f"Script is too long (max {config.MAX_SCRIPT_CHARS} characters).")
    if not (config.SPEED_MIN <= speed <= config.SPEED_MAX):
        raise HTTPException(422, f"Speed must be between {config.SPEED_MIN} and {config.SPEED_MAX}.")
    if not (config.EXAGGERATION_MIN <= exaggeration <= config.EXAGGERATION_MAX):
        raise HTTPException(
            422, f"Expressiveness must be between {config.EXAGGERATION_MIN} and {config.EXAGGERATION_MAX}."
        )

    reference_path = _voice_reference_path(voice)
    sentences = _split_sentences(text)

    chatterbox = get_chatterbox_service()
    t0 = time.perf_counter()
    clips = []
    sr = None
    try:
        for sentence in sentences:
            wav_np, sample_rate = chatterbox.generate(
                sentence, str(reference_path), language=language, exaggeration=exaggeration
            )
            sr = sample_rate
            clips.append(wav_np)
    except ChatterboxError as e:
        raise HTTPException(502, f"Speech generation failed: {e}") from e
    t_generate = time.perf_counter() - t0

    if len(clips) > 1:
        gap = np.zeros(int(_SILENCE_GAP_SECONDS * sr), dtype=clips[0].dtype)
        pieces = []
        for i, clip in enumerate(clips):
            pieces.append(clip)
            if i < len(clips) - 1:
                pieces.append(gap)
        final_wav = np.concatenate(pieces)
    else:
        final_wav = clips[0]

    # Chatterbox has no native speech-rate control -- pace is a pitch-preserving
    # time-stretch (phase vocoder) applied to the finished clip instead. Skip it
    # at speed=1.0 exactly: time_stretch is a no-op there but not a free one.
    if speed != 1.0:
        final_wav = librosa.effects.time_stretch(final_wav.astype(np.float32), rate=speed)

    audio_id = f"{voice}_{uuid.uuid4().hex[:8]}.wav"
    audio_path = config.OUTPUT_DIR / audio_id
    sf.write(str(audio_path), final_wav, sr, subtype="PCM_16")

    duration_s = len(final_wav) / sr
    logger.info(
        "Generated %d sentence(s), %.1fs of audio, in %.1fs (voice=%s)",
        len(sentences), duration_s, t_generate, voice,
    )

    return {
        "audio_url": f"/api/audio/{audio_id}",
        "sentence_count": len(sentences),
        "duration_seconds": round(duration_s, 2),
        "generation_seconds": round(t_generate, 2),
        "speed": speed,
        "exaggeration": exaggeration,
    }


@router.get("/api/audio/{filename}")
async def get_audio(filename: str):
    safe_name = Path(filename).name  # filenames are server-generated; still guard against path traversal
    file_path = config.OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "Audio not found.")
    return FileResponse(str(file_path), media_type="audio/wav")
