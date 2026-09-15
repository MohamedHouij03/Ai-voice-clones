# 🇫🇷 AI Teacher Voice Suite

**A demo prototype built to introduce to [Dreaming French](https://dreaming.com/french)** — a French-learning platform teaching through comprehensible input rather than drills. Not yet an official collaboration: this is a working prototype, built with real teachers' consented voices, meant to be pitched to the platform directly.

This project trains AI clones of consenting Dreaming French teachers' voices and puts them to work two ways: a **real-time conversational practice partner** students can actually talk to, and an **instant narration generator** that lets editors patch a missed or flubbed line in a teacher's own voice without a re-shoot.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="Chatterbox" src="https://img.shields.io/badge/Chatterbox_Multilingual-V3-8A2BE2" />
  <img alt="Whisper" src="https://img.shields.io/badge/faster--whisper-STT-4B8BBE" />
  <img alt="RAG" src="https://img.shields.io/badge/semantic_search-RAG-FF6F00" />
  <img alt="Status" src="https://img.shields.io/badge/status-pitch_prototype-yellow" />
  <img alt="License" src="https://img.shields.io/badge/license-private_project-lightgrey" />
</p>

---

## 🎥 Demo Video

[![Demo Video](https://img.youtube.com/vi/fajR3uJ5tgQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=fajR3uJ5tgQ)

---

## 📖 Table of Contents

- [Why this exists](#-why-this-exists)
- [The two tools](#-the-two-tools)
- [Meet the cloned teachers](#-meet-the-cloned-teachers)
- [How it works](#-how-it-works)
- [Retrieval-Augmented video matching (RAG)](#-retrieval-augmented-video-matching-rag)
- [Tech stack](#-tech-stack)
- [Project structure](#-project-structure)
- [Getting started](#-getting-started)
- [Voice cloning consent — read first](#-voice-cloning-consent--read-first)
- [Roadmap](#-roadmap)

---

## 🎯 Why this exists

Dreaming French teaches through **comprehensible input** — real teachers, real conversation, no drilling. Two everyday problems this project was built to solve:

1. **Students want to practice speaking, not just watch videos.** A cloned teacher voice that can hold a real, level-appropriate conversation gives students a low-stakes way to speak French between lessons — in a voice they already know and trust.
2. **Video editors lose time to re-shoots.** When a teacher flubs a line, forgets a word, or a sentence needs a small fix after filming, the whole thing traditionally means booking the teacher again. Generating that one missing line in their own cloned voice turns a re-shoot into a 30-second fix.

Both problems share the same underlying capability — an accurate, natural-sounding clone of each teacher's real voice — so both tools were built on the same voice-cloning core.

## 🧩 The two tools

### 1. AI Assistant for Practice
A real-time voice conversation with an AI clone of a Dreaming French teacher.

- Push-to-talk mic input, transcribed locally, answered by an LLM, spoken back in the teacher's cloned voice
- Follows Dreaming's own teaching method — **comprehensible input** (Krashen), not quizzes or grammar drills, calibrated to the student's level (superbeginner → advanced)
- Each teacher offers **their own conversation topics**, matched to their real personality and interests (e.g. Audrey → photography and French cities, Chloé → gaming and streaming)
- Picking a topic starts a **pre-scripted vocabulary quiz** for that subject — instant, one-word-answer questions with zero LLM latency, since the content is pre-written rather than generated live
- Finishing a topic surfaces that teacher's **own real videos** from the Dreaming French catalog, matched via semantic search (see [RAG](#-retrieval-augmented-video-matching-rag))
- A second mode in the same app, **"Générer des phrases,"** is the same narration-generation capability as the standalone tool below, available without leaving a practice session

### 2. AI Sentence Generator
A standalone tool for video editors: paste a script, pick a teacher's cloned voice, get back a narration WAV.

- No speech recognition, no LLM — you supply the exact text you want spoken
- Adjustable **pace** (0.75×–1.3×, pitch-preserving time-stretch) and **expressiveness** (Chatterbox's emotional-intensity parameter), so a generated line can be tuned to match the surrounding footage
- Every script, however long, comes back as one clean, downloadable WAV
- Deliberately simple — no sessions, no caching, no streaming — built for producing a handful of narration clips per video, not real-time conversation

## 👥 Meet the cloned teachers

Eight teacher voices are cloned and available across both tools today: **Amanda, Clément, Chloé, Line, Guénaël, Audrey, Mélisande,** and **Olivia** — each trained from a short, consented reference recording and cleaned via automatic loudness normalization and best-take window selection before cloning.

## ⚙️ How it works

```
  Student speaks  ──▶  faster-whisper (local STT)  ──▶  LLM reply (Gemini/OpenAI/Anthropic)
                                                                │
                                                                ▼
  Cloned teacher voice  ◀──  Chatterbox Multilingual V3 (voice cloning TTS)
```

- **STT** runs locally on CPU (faster-whisper), keeping the GPU free for voice cloning.
- **LLM** is provider-agnostic — Gemini is the default (free tier), with OpenAI and Anthropic supported behind the same interface.
- **TTS** is Chatterbox Multilingual V3, loaded once and kept resident on GPU for the life of the server.
- Replies are streamed sentence-by-sentence, with LLM generation and speech synthesis overlapping in a producer/consumer pipeline, so the student hears the first words while the rest is still being generated.
- Common openers, small talk, and full topic-quiz content are **pre-synthesized and cached**, skipping the LLM and the GPU entirely for anything predictable.

## 🔍 Retrieval-Augmented video matching (RAG)

The "recommend a relevant video" feature uses real retrieval-augmented generation, not just keyword search:

- Every catalog video's title + description is embedded once with a local, offline multilingual sentence-embedding model (no external API, no extra cost per query).
- A conversation topic (or a student's own words) is embedded the same way, and the closest video by cosine similarity is surfaced.
- This replaced an earlier keyword-only (FTS5) search, which regularly missed matches — most of the catalog's titles are in English while students speak French, so a query like *"j'adore le vin"* would never keyword-match a video titled *"Why French Wine Tasting Is Bullsh\*t."* Semantic embeddings catch that match directly.
- The video catalog itself is sourced entirely from **YouTube's official Data API v3** — not scraped — out of respect for Dreaming French's own site policies.

## 🛠️ Tech stack

| Layer | Technology | Why |
|---|---|---|
| Voice cloning | **Chatterbox Multilingual V3** (Resemble AI) | Best open, GPU-runnable multilingual voice-cloning TTS available; installed from source for true V3 support |
| Speech-to-text | **faster-whisper** | Fast, accurate, fully local — no audio ever leaves the machine |
| LLM | **Gemini** (default) / OpenAI / Anthropic | Swappable behind one interface; Gemini chosen as default for its free tier |
| Semantic search | Local multilingual sentence-embedding model | Real RAG for video recommendations, zero extra API cost |
| Backend | **FastAPI** + Uvicorn | Async streaming responses, clean typed routes |
| Frontend | Vanilla JS/HTML/CSS | No framework overhead for a tightly-scoped UI |
| Audio processing | **librosa**, pyloudnorm, soundfile | Reference-clip cleanup, loudness normalization, pitch-preserving pace control |
| Storage | SQLite | Lightweight local video catalog |
| Video catalog | **YouTube Data API v3** | Official, ToS-compliant source for Dreaming French's own channel |

## 📁 Project structure

```
├── french-teacher/      # AI Assistant for Practice (+ in-app sentence generation)
└── voice-generator/     # Standalone AI Sentence Generator for video editors
```

> **Note:** both tools currently live together **for demonstration/pitch purposes only**. They share the same voice-cloning approach and, originally, the same codebase — but each already has its own independent `voices/` folder, its own server, and its own README, so they run and deploy independently today. If Dreaming French adopts this, they'll be **split into two separate projects** for delivery.

## 🚀 Getting started

Each tool runs independently and has its own full setup guide:

- [`french-teacher/README.md`](./french-teacher/README.md) — the practice assistant (port `8000`)
- [`voice-generator/README.md`](./voice-generator/README.md) — the sentence generator (port `8100`)

Both need a CUDA-capable GPU for Chatterbox and share the same install pattern (pinned CUDA PyTorch first, then Chatterbox from source). **Run only one at a time** — both are GPU-heavy and will contend for VRAM if run together.

## 🔒 Voice cloning consent — read first

Every cloned voice in this project belongs to a real person who has explicitly authorized their voice to be used this way. No recording of anyone who has not consented is included or should be added. This project must never be used to impersonate someone without authorization.

## 🗺️ Roadmap

- [ ] Record and embed the demo video above
- [ ] Introduce and pitch this to Dreaming French
- [ ] Expand pre-scripted quiz content per topic
- [ ] Extend video recommendations to more teachers
- [ ] If adopted, split into two independently deployable repositories for delivery
