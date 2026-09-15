"""
Local, offline text embeddings for semantic video search (see
find_relevant_video in video_library.py) -- replaces that function's earlier
FTS5 keyword-match approach, which missed near-matches sharing no exact word
(e.g. "je veux visiter Paris" vs a video titled "une journée à Paris").

Uses transformers directly (already a dependency for Chatterbox) rather than
adding the sentence-transformers package: the configured model
(config.VIDEO_EMBEDDING_MODEL) is just BERT/XLM-R-family weights, loadable
with plain AutoTokenizer/AutoModel -- sentence-transformers only adds a thin
convenience wrapper around mean-pooling, reimplemented here in a few lines to
avoid a new dependency with its own version-compatibility risk (see
requirements.txt's notes on this project's history of chasing down Python
3.14 wheel availability for every dependency).

CPU by default, same reasoning as WHISPER_DEVICE: the GPU is reserved for
Chatterbox, and embedding a handful of short title/description strings is
fast enough on CPU that the GPU contention isn't worth risking.
"""
import logging
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .. import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tokenizer = None
_model = None


def load_model() -> None:
    """Loads the tokenizer/model once and keeps them resident -- called
    eagerly at server startup (see main.py's lifespan) alongside Chatterbox/
    Whisper so the first real request doesn't pay a multi-second model-load
    cost. Safe to call more than once (no-ops after the first load)."""
    global _tokenizer, _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        logger.info("Loading video embedding model %s on cpu", config.VIDEO_EMBEDDING_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(config.VIDEO_EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(config.VIDEO_EMBEDDING_MODEL)
        _model.eval()
        logger.info("Video embedding model loaded.")


def embed_text(text: str) -> np.ndarray:
    """Returns a unit-normalized embedding vector (float32) for `text`,
    mean-pooled over token embeddings (masked by attention_mask) -- the same
    pooling recipe sentence-transformers itself uses for this model family.
    Unit-normalized so cosine similarity reduces to a plain dot product at
    search time (see video_library.find_relevant_video)."""
    load_model()
    with torch.no_grad():
        encoded = _tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
        output = _model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).squeeze(0)
        normalized = pooled / pooled.norm(p=2).clamp(min=1e-9)
    return normalized.numpy().astype(np.float32)
