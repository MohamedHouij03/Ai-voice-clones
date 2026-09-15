"""
Central configuration, loaded from environment variables / .env.
Nothing here should be hardcoded again elsewhere in the backend -- add new
tunables here and read them from `config` at the call site.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "conversation_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Active teacher (folder name under voices/, e.g. "teacher_1") ---
ACTIVE_TEACHER_VOICE = os.getenv("ACTIVE_TEACHER_VOICE", "teacher_1")

# --- Chatterbox (see scripts/phase1_poc.py for how these were chosen/verified) ---
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "auto")  # "auto" | "cuda" | "cpu"
CHATTERBOX_T3_MODEL = os.getenv("CHATTERBOX_T3_MODEL", "v3")
CHATTERBOX_LANGUAGE = os.getenv("CHATTERBOX_LANGUAGE", "fr")
CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5"))
CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))

# --- STT (faster-whisper) ---
# Default device is CPU deliberately: Chatterbox is loaded persistently on the
# 8GB GPU, and short student utterances transcribe fast enough on CPU with an
# int8 "small" model that it isn't worth the VRAM contention. Override via env
# if you want Whisper on GPU too and have headroom.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- LLM (see backend/services/llm_service.py for the provider abstraction) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# --- Session / conversation ---
# Matches Dreaming's own level scheme (see e.g. app.dreaming.com/french/browse
# ?level=superbeginner|beginner|intermediate|advanced in their sitemap) rather
# than CEFR, for consistency with the rest of this app's Dreaming-matched UI.
DEFAULT_STUDENT_LEVEL = os.getenv("DEFAULT_STUDENT_LEVEL", "beginner")
VALID_LEVELS = ("superbeginner", "beginner", "intermediate", "advanced")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # 25MB safety cap

# --- Session cleanup ---
# A session idle longer than this (no turns added) is swept by the background
# cleanup task -- fallback for browsers that never send the pagehide beacon
# (crash, force-quit, etc). See conversation_service.cleanup_expired_sessions().
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(30 * 60)))  # 30 min
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", str(5 * 60)))  # 5 min

# --- Common-phrase audio cache (see services/common_phrases.py) ---
COMMON_PHRASES_PATH = PROJECT_ROOT / "backend" / "data" / "common_phrases.json"

# --- Streaming TTS chunking (see conversation_service._stream_chunks) ---
# A completed sentence (ends in .!?) always becomes a chunk regardless of length.
# For a sentence that keeps growing without ending, MAX forces an earlier split at
# the best available comma/semicolon/colon once the buffer passes this length, so
# one long sentence can't single-handedly delay Time To First Audio. MIN prevents
# splitting so early that a chunk would be an unnaturally short fragment.
MIN_TTS_CHUNK_CHARS = int(os.getenv("MIN_TTS_CHUNK_CHARS", "20"))
MAX_TTS_CHUNK_CHARS = int(os.getenv("MAX_TTS_CHUNK_CHARS", "120"))

# --- LLM reply length ---
# TTS dominates turn latency (see README), so a shorter reply is a direct,
# reliable latency win on top of the "1-3 short sentences" prompt guidance --
# this is a hard cap, the prompt is the soft guidance. ~140 tokens is roughly
# 2-3 short French sentences; raise it if replies start truncating mid-thought.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "140"))

# --- Access control (see main.py's basic_auth_gate middleware) ---
# Empty by default (no auth) for local dev. When hosting this publicly, set
# both in .env -- the middleware only activates once APP_PASSWORD is non-empty,
# so a forgotten/blank password fails CLOSED to "no auth" rather than locking
# everyone out, but that also means you MUST set this before exposing the app
# to the internet or it is wide open (anyone can burn your GPU/LLM budget).
APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# --- Direct sentence generation ("Générer des phrases" mode -- see
# backend/services/sentence_generation.py) ---
# Separate from OUTPUT_DIR (conversation audio, swept by the session-TTL
# cleanup sweep): these files are meant to be downloaded and kept by the
# user for video editing, not auto-deleted after 30 minutes.
GENERATED_AUDIO_DIR = PROJECT_ROOT / "output" / "generated_sentences"
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- Video references (see services/video_library.py) ---
# Sourced from YouTube's official Data API v3 (scripts/fetch_youtube_channel.py),
# not scraped from Dreaming's own app -- see that script's docstring for why.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
# Dreaming French's own channel, per https://www.youtube.com/channel/UCG7lancLEOKXZ7lEEyzvwjA
DREAMING_YOUTUBE_CHANNEL_ID = os.getenv("DREAMING_YOUTUBE_CHANNEL_ID", "UCG7lancLEOKXZ7lEEyzvwjA")
VIDEO_LIBRARY_DB_PATH = PROJECT_ROOT / "backend" / "data" / "videos.db"

# --- Video semantic matching (RAG-style retrieval -- see
# services/video_embeddings.py) ---
# A small multilingual sentence-embedding model, loaded via plain
# transformers (already a dependency for Chatterbox) rather than the
# sentence-transformers package -- see that module's docstring for why.
VIDEO_EMBEDDING_MODEL = os.getenv("VIDEO_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Cosine similarity floor below which find_relevant_video returns None rather
# than the closest-of-a-bad-bunch match -- a vector search always returns
# SOME nearest vector even for a wildly unrelated query, unlike the old FTS5
# MATCH (which naturally returned zero rows when nothing shared a keyword).
VIDEO_MATCH_MIN_SIMILARITY = float(os.getenv("VIDEO_MATCH_MIN_SIMILARITY", "0.25"))
