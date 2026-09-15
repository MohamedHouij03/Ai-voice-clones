# Voice Generator

A standalone tool for video editors: paste a script, pick a cloned voice, get
back a narration WAV. No speech recognition, no LLM -- you supply the exact
text you want spoken.

This is a sibling project to `../french-teacher/` (the conversational French
practice app). Both were split out of a single original project so each can
be developed, run, and deployed independently. They share the same
voice-cloning technology (Chatterbox Multilingual V3) and, as of the split,
the same starting set of cloned voices -- but **the `voices/` folders are
independent copies, not shared**. Adding, removing, or re-recording a voice
in one project does not affect the other; do it in both if you want them to
stay in sync (or delete this note once that stops mattering to you).

## Voice cloning consent (read first)

Every reference recording under `voices/` must belong to a real person who
has explicitly authorized their voice to be cloned for this tool. Do not add
a recording of anyone who has not consented to this use, and do not use this
project to impersonate someone without authorization.

## Setup

```powershell
cd "voice-generator"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1. CUDA-enabled PyTorch FIRST, pinned (see requirements.txt for why order matters)
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt

# 3. Chatterbox itself, from source (see requirements.txt for why)
pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2" --no-deps
```

Copy `.env.example` to `.env` if you want to change any defaults (device,
exaggeration, cfg weight, max script length).

## Running

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8100
```

Open http://127.0.0.1:8100 -- pick a voice, paste your script, adjust pace
(0.75x-1.3x) and expressiveness if needed, click Generate, then play or
download the result. First startup loads Chatterbox onto the GPU, which
takes a while the first time (weights are cached in `~/.cache/huggingface`
afterward, shared with the french-teacher project).

Note the port (`8100`) differs from french-teacher's default (`8000`) so
both can run at the same time on one machine -- **but Chatterbox is
GPU-heavy, and running two instances of it on the same 8GB GPU at once will
badly contend with each other and slow both down.** Run one at a time unless
you know you have the VRAM headroom.

## Adding a voice

```
voices/<id>/
  raw.wav          # the original recording, as supplied
  reference.wav    # cleaned version, generate with the command below
  meta.json         # {"display_name": "Whatever you want shown in the picker"}
```

```powershell
python scripts/prepare_reference.py --input voices/my_voice/raw.wav --output voices/my_voice/reference.wav
```

The picker on the page auto-discovers any folder under `voices/` that has a
`reference.wav`.

## How it works

- `backend/chatterbox_service.py` -- loads the model once at startup, keeps
  it resident on the GPU. Identical to the same file in `../french-teacher/`.
- `backend/api.py` -- `POST /api/generate` splits the script into sentences,
  synthesizes each one, and stitches them back into a single WAV (with a
  short silence gap between sentences) so you get one clean downloadable
  file regardless of script length, instead of hitting Chatterbox's
  per-call length limits on a long script. Pace is applied afterward as a
  pitch-preserving time-stretch (Chatterbox has no native speech-rate
  control); expressiveness maps directly to Chatterbox's own `exaggeration`
  parameter.
- No caching, no sessions, no streaming -- every generation is a plain,
  blocking request that returns a complete file. That's the right amount of
  complexity for a tool used to produce a handful of narration clips per
  video, as opposed to the french-teacher project's real-time conversational
  use case.
- Generated files land in `output/` and are **not** automatically deleted --
  they're the actual thing you're here to produce, so clean that folder out
  yourself whenever you like.
