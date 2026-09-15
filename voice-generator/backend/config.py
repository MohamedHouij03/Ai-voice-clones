"""
Central configuration, loaded from environment variables / .env.
Nothing here should be hardcoded again elsewhere -- add new tunables here
and read them from `config` at the call site.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Chatterbox (see the ai-french-teacher project's scripts/phase1_poc.py
# for how these were originally chosen/verified -- this project reuses the
# same voice-cloning pipeline, just without STT/LLM, for generating narration
# clips from arbitrary script text) ---
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "auto")  # "auto" | "cuda" | "cpu"
CHATTERBOX_T3_MODEL = os.getenv("CHATTERBOX_T3_MODEL", "v3")
CHATTERBOX_LANGUAGE = os.getenv("CHATTERBOX_LANGUAGE", "fr")
CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5"))
CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))

MAX_SCRIPT_CHARS = int(os.getenv("MAX_SCRIPT_CHARS", "5000"))  # safety cap on pasted script length

# --- Speaking pace (see api.py's time-stretch step) ---
# Chatterbox has no native speech-rate control, so pace is applied as a
# pitch-preserving time-stretch (librosa) on the finished audio, not passed
# into the model itself. 1.0 = as generated; bounds are generous but finite
# so a stray value can't produce a garbled/unusable clip.
SPEED_MIN = float(os.getenv("SPEED_MIN", "0.7"))
SPEED_MAX = float(os.getenv("SPEED_MAX", "1.4"))

# --- Expressiveness (Chatterbox's own "exaggeration" parameter) ---
# Unlike pace, this IS a native Chatterbox generation parameter -- it doesn't
# specifically target punctuation, but higher values make the model's
# prosody more emotionally varied/pronounced (which is the closest lever
# available to "more noticeable question/comma intonation"). CHATTERBOX_
# EXAGGERATION above stays the *default* pre-fill; bounds keep a stray value
# from pushing the model into unstable/over-the-top territory.
EXAGGERATION_MIN = float(os.getenv("EXAGGERATION_MIN", "0.3"))
EXAGGERATION_MAX = float(os.getenv("EXAGGERATION_MAX", "1.0"))
