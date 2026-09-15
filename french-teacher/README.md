# AI French Teacher

A conversational French tutor: speak into your microphone, get transcribed locally (faster-whisper), get a reply from an LLM playing a specific teacher's persona, and hear it spoken back in that teacher's own cloned voice (Chatterbox Multilingual V3).

Teaching follows a comprehensible-input approach (real conversation calibrated to the student's level, not quizzes or grammar drills) rather than testing — matching how Dreaming French itself teaches.

This is a sibling project to `../voice-generator/` (a standalone narration tool for video editors). Both share the same voice-cloning approach, but **`voices/` is an independent copy in each project, not shared** — adding a voice here does not add it there, and vice versa.

## Features

- **Real-time voice conversation** — push-to-talk mic input, streamed replies (audio starts playing before the full reply finishes generating)
- **Level-aware** — superbeginner through advanced, each with its own vocabulary/pacing guidance
- **Per-teacher topics** — each teacher offers conversation topics matched to their own personality and interests; picking one starts an instant, pre-scripted vocabulary quiz (no LLM call, no wait)
- **Video recommendations** — finishing a topic surfaces that teacher's own videos from the Dreaming French catalog, matched by semantic search rather than keyword overlap
- **Générer des phrases** — a second mode for typing an exact sentence and getting it back in any teacher's voice, useful for patching a line without leaving the app
- **Response caching** — predictable lines (greetings, small talk, quiz content) are pre-synthesized and cached per teacher, skipping both the LLM and Chatterbox on repeat hits

## Voice cloning consent — read first

Every reference recording under `voices/` must belong to a real person who has explicitly authorized their voice to be cloned for this tool. Do not add a recording of anyone who has not consented to this use, and do not use this project to impersonate someone without authorization.

## Setup

```powershell
cd "french-teacher"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1. CUDA-enabled PyTorch FIRST, pinned (see requirements.txt for why order matters)
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt

# 3. Chatterbox itself, from source — PyPI's release predates Multilingual V3 support
pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2" --no-deps
```

Copy `.env.example` to `.env` and add a free [Gemini API key](https://aistudio.google.com/apikey) (the default LLM provider, chosen for its free tier — OpenAI and Anthropic are also supported behind the same interface, see `backend/services/llm_service.py`).

## Running

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, allow microphone access, pick a teacher and level, and press the mic button to talk. First startup loads Chatterbox onto the GPU and downloads the Whisper model — this takes a few minutes the first time, seconds after that (models are cached).

**Chatterbox is GPU-heavy** — avoid running this alongside `../voice-generator/` (or any other process holding the model) on the same GPU, since two instances contending for VRAM will badly slow both down.

## Adding a teacher voice

```
voices/<id>/
  raw.wav          # the original recording, as supplied
  reference.wav    # cleaned version — see the command below
  meta.json        # {"display_name": "Whatever you want shown in the picker"}
  topics.json       # optional: this teacher's conversation topics + quiz content
```

```powershell
python scripts/prepare_reference.py --input voices/my_voice/raw.wav --output voices/my_voice/reference.wav
```

The app auto-discovers any folder under `voices/` that has a `reference.wav` — adding a teacher is a data operation, not a code change. `scripts/precompute_canned_audio.py` can pre-warm a new teacher's greeting/small-talk/quiz cache ahead of time instead of paying that cost on the first real conversation.
