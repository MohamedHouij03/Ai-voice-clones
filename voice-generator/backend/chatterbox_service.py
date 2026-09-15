"""
Loads Chatterbox Multilingual V3 once and keeps it resident on the GPU (or CPU
fallback). Ported from the ai-french-teacher project's
backend/services/chatterbox_service.py -- see that project's scripts/phase1_poc.py
and README for why Chatterbox is installed from GitHub source rather than
PyPI (PyPI's chatterbox-tts release predates V3/`t3_model` support).
"""
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from . import config

logger = logging.getLogger(__name__)


class ChatterboxError(Exception):
    """Raised for any Chatterbox failure; the API layer turns this into a clean HTTP error."""


class ChatterboxService:
    def __init__(self):
        self._model = None
        self.device: str = ""

    def load_model(self) -> None:
        if self._model is not None:
            return
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if config.CHATTERBOX_DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.CHATTERBOX_DEVICE

        logger.info(
            "Loading ChatterboxMultilingualTTS (t3_model=%s) on device=%s",
            config.CHATTERBOX_T3_MODEL,
            self.device,
        )
        try:
            self._model = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device, t3_model=config.CHATTERBOX_T3_MODEL
            )
        except Exception as e:
            raise ChatterboxError(f"Failed to load Chatterbox model: {e}") from e
        logger.info("Chatterbox model loaded and resident on %s.", self.device)

    @property
    def sample_rate(self) -> int:
        self.load_model()
        return self._model.sr

    def generate(
        self, text: str, reference_audio: str, language: str = None, exaggeration: float = None
    ) -> Tuple[np.ndarray, int]:
        self.load_model()

        reference_path = Path(reference_audio)
        if not reference_path.exists():
            raise ChatterboxError(f"Voice reference not found: {reference_path}")
        if not text or not text.strip():
            raise ChatterboxError("Cannot synthesize empty text.")

        language = language or config.CHATTERBOX_LANGUAGE
        exaggeration = config.CHATTERBOX_EXAGGERATION if exaggeration is None else exaggeration
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        try:
            with autocast_ctx:
                wav = self._model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=str(reference_path),
                    exaggeration=exaggeration,
                    cfg_weight=config.CHATTERBOX_CFG_WEIGHT,
                )
        except torch.cuda.OutOfMemoryError as e:
            raise ChatterboxError(
                "GPU ran out of memory during speech generation. Try shorter text."
            ) from e
        except Exception as e:
            raise ChatterboxError(f"Speech generation failed: {e}") from e

        wav_np = wav.squeeze(0).detach().cpu().numpy()
        return wav_np, self._model.sr


_service = ChatterboxService()


def get_chatterbox_service() -> ChatterboxService:
    return _service
