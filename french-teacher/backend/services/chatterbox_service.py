"""
Loads Chatterbox Multilingual V3 once and keeps it resident on the GPU (or CPU
fallback). See scripts/phase1_poc.py for the proof-of-concept this was extracted
from -- including why Chatterbox is installed from GitHub source rather than
PyPI (PyPI's chatterbox-tts release predates V3/`t3_model` support), and why no
pace/time-stretch post-processing is applied here: it measurably hurt audio
quality in testing, so live conversation uses the model's raw, natural output.
"""
import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .. import config

logger = logging.getLogger(__name__)


class ChatterboxError(Exception):
    """Raised for any Chatterbox failure; the API layer turns this into a clean HTTP error."""


class ChatterboxService:
    def __init__(self):
        self._model = None
        self.device: str = ""
        # Serializes ALL calls into the model, across every request thread. Two
        # reasons: (1) the GPU-contention finding this project's own testing
        # turned up -- concurrent Chatterbox jobs on one GPU are dramatically
        # slower than sequential ones, not faster; (2) self._prepared_key/
        # self._model.conds below is mutable model state shared across threads,
        # so two calls for two different teachers interleaving would corrupt
        # each other's in-flight generation.
        self._lock = threading.Lock()
        # Tracks which (reference_path, exaggeration) self._model.conds currently
        # reflects, so repeat calls for the SAME teacher can skip re-running
        # prepare_conditionals() -- see the comment inside generate() for why
        # this matters (it turned out to dominate per-call latency).
        self._prepared_key = None

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

    def generate(self, text: str, reference_audio: str, language: str = None) -> Tuple[np.ndarray, int]:
        self.load_model()

        reference_path = Path(reference_audio)
        if not reference_path.exists():
            raise ChatterboxError(f"Teacher reference voice not found: {reference_path}")
        if not text or not text.strip():
            raise ChatterboxError("Cannot synthesize empty text.")

        language = language or config.CHATTERBOX_LANGUAGE
        exaggeration = config.CHATTERBOX_EXAGGERATION
        # autocast(bf16) was added on the theory that the T3 checkpoint's fp32
        # weights were leaving GPU tensor-core speed on the table. Profiling
        # this machine during generation (nvidia-smi during an active call)
        # showed only ~5% GPU utilization and a handful of watts of power draw
        # throughout -- i.e. this workload is NOT compute-bound here, it's
        # bound by CPU-side/kernel-launch overhead between the ~1000 tiny
        # autoregressive sampling steps. Measured A/B on this machine: disabling
        # autocast raised the sampling rate from ~4 it/s to ~5.2 it/s (autocast's
        # per-op dtype-cast overhead was pure cost with no compute win to offset
        # it here). Left as a no-op context rather than deleted in case this
        # ever runs on a setup where the workload genuinely is compute-bound.
        autocast_ctx = nullcontext()
        with self._lock:
            try:
                with autocast_ctx:
                    # Chatterbox's own generate() unconditionally re-runs
                    # prepare_conditionals() (load + resample the reference wav,
                    # a voice-encoder forward pass, s3gen ref conditioning) any
                    # time audio_prompt_path is passed. That's avoidable work on
                    # every call after the first for the SAME teacher within a
                    # turn (and this project's chunking splits one reply into
                    # several short calls specifically to lower TTFA -- paying
                    # this cost per chunk would fight that directly), so: only
                    # pass audio_prompt_path when the (teacher, exaggeration)
                    # pair actually changed since the last call; otherwise reuse
                    # self._model.conds via audio_prompt_path=None, which
                    # Chatterbox's own generate() supports natively. NOTE: this
                    # is a real but comparatively small win -- profiling on this
                    # machine (see autocast_ctx above) found the sampling loop
                    # itself, not conditioning prep, is the dominant cost; a
                    # kernel-launch-overhead-bound autoregressive loop like this
                    # would benefit far more from torch.compile/CUDA graphs,
                    # which is future work, not part of this change.
                    key = (str(reference_path), exaggeration)
                    prompt_path = None
                    if key != self._prepared_key:
                        prompt_path = str(reference_path)

                    wav = self._model.generate(
                        text,
                        language_id=language,
                        audio_prompt_path=prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=config.CHATTERBOX_CFG_WEIGHT,
                    )
                    self._prepared_key = key
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
