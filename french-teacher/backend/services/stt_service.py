"""
Speech-to-text via faster-whisper, loaded once at startup. Accepts whatever
container the browser's MediaRecorder produces (webm/opus) directly -- faster-
whisper decodes audio via PyAV, so no separate ffmpeg install is required.
"""
import logging

from .. import config

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Raised for any STT failure; the API layer turns this into a clean HTTP error."""


class STTService:
    def __init__(self):
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        try:
            self._model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {e}") from e
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_path: str, language: str = "fr") -> str:
        self.load_model()
        try:
            segments, _info = self._model.transcribe(
                audio_path, language=language, beam_size=5, vad_filter=True
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e
        return text


_service = STTService()


def get_stt_service() -> STTService:
    return _service
