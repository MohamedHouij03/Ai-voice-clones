"""
Local SQLite store of Dreaming French's YouTube video catalog, populated by
scripts/fetch_youtube_channel.py (via YouTube's official Data API v3 -- see
that script's docstring for why this project uses that instead of scraping
Dreaming's own app). Lets the conversation teacher occasionally reference a
real, relevant video by title + link for a topic that came up.

find_relevant_video() matches by semantic similarity (see
services/video_embeddings.py -- a small local, offline embedding model, no
external API) rather than keyword overlap: each video's title+description is
embedded once and cached in the `embedding` column, the query text is
embedded the same way, and the closest video by cosine similarity wins. This
replaced an earlier FTS5 keyword-MATCH approach that missed near-matches
sharing no exact word (e.g. "je veux visiter Paris" vs a video titled "une
journée à Paris").
"""
import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL,
    published_at TEXT,
    thumbnail_url TEXT,
    channel_title TEXT,
    embedding BLOB
);
"""


@dataclass
class VideoRef:
    video_id: str
    title: str
    description: str
    url: str
    thumbnail_url: str = ""


def _connect() -> sqlite3.Connection:
    config.VIDEO_LIBRARY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(config.VIDEO_LIBRARY_DB_PATH))
    con.executescript(_SCHEMA)
    try:
        con.execute("ALTER TABLE videos ADD COLUMN embedding BLOB")
    except sqlite3.OperationalError:
        pass  # already migrated -- CREATE TABLE IF NOT EXISTS above doesn't add columns to an existing table
    return con


def upsert_video(
    video_id: str,
    title: str,
    description: str,
    url: str,
    published_at: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    channel_title: Optional[str] = None,
) -> None:
    # A re-synced video's title/description may have changed -- clear any
    # cached embedding so find_relevant_video recomputes it from the new
    # text instead of matching on stale content.
    with _connect() as con:
        con.execute(
            """
            INSERT INTO videos (video_id, title, description, url, published_at, thumbnail_url, channel_title)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                url=excluded.url,
                published_at=excluded.published_at,
                thumbnail_url=excluded.thumbnail_url,
                channel_title=excluded.channel_title,
                embedding=NULL
            """,
            (video_id, title, description, url, published_at, thumbnail_url, channel_title),
        )
    _embedding_cache.pop(video_id, None)


def count_videos() -> int:
    with _connect() as con:
        return con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


# video_id -> unit-normalized embedding (np.float32). Populated once per
# process by _ensure_embeddings_loaded and reused for every query after
# that -- the catalog is small (dozens of videos) and changes only via the
# offline fetch script, so there's no need to re-read/re-embed per request.
_embedding_cache: dict = {}


def _ensure_embeddings_loaded(con: sqlite3.Connection) -> None:
    from . import video_embeddings  # local import: avoids loading torch/transformers just to open the DB

    rows = con.execute("SELECT video_id, title, description, embedding FROM videos").fetchall()
    to_store = []
    for video_id, title, description, embedding_blob in rows:
        if video_id in _embedding_cache:
            continue
        if embedding_blob:
            _embedding_cache[video_id] = np.frombuffer(embedding_blob, dtype=np.float32)
            continue
        vector = video_embeddings.embed_text(f"{title}. {description}".strip())
        _embedding_cache[video_id] = vector
        to_store.append((vector.tobytes(), video_id))

    if to_store:
        con.executemany("UPDATE videos SET embedding = ? WHERE video_id = ?", to_store)
        logger.info("Computed and cached embeddings for %d video(s).", len(to_store))


def warm_embeddings() -> None:
    """Backfills any missing video embeddings up front -- called at server
    startup (see main.py's lifespan) alongside Chatterbox/Whisper so the
    first real "as-tu une vidéo ?" doesn't pay a multi-second embedding cost
    mid-conversation. Safe/cheap to call again later (a no-op once every row
    already has a cached embedding)."""
    with _connect() as con:
        _ensure_embeddings_loaded(con)


def find_relevant_video(topic_text: str) -> Optional[VideoRef]:
    """Best-matching video for `topic_text` (e.g. recent conversation
    content) by semantic similarity over title+description, or None if the
    catalog is empty or nothing is similar enough (see
    config.VIDEO_MATCH_MIN_SIMILARITY)."""
    from . import video_embeddings

    query_text = topic_text.strip()
    if not query_text:
        return None

    with _connect() as con:
        _ensure_embeddings_loaded(con)
        rows = {
            r[0]: r for r in con.execute(
                "SELECT video_id, title, description, url, thumbnail_url FROM videos"
            ).fetchall()
        }
    if not rows:
        return None

    query_vector = video_embeddings.embed_text(query_text)
    best_id, best_score = None, -1.0
    for video_id, vector in _embedding_cache.items():
        if video_id not in rows:
            continue
        score = float(np.dot(query_vector, vector))  # both unit-normalized -> dot product == cosine similarity
        if score > best_score:
            best_id, best_score = video_id, score

    if best_id is None or best_score < config.VIDEO_MATCH_MIN_SIMILARITY:
        return None

    row = rows[best_id]
    return VideoRef(video_id=row[0], title=row[1], description=row[2], url=row[3], thumbnail_url=row[4] or "")


def get_videos_by_ids(video_ids: list) -> list:
    """Resolves a list of catalog video ids (e.g. from a teacher's own
    voices/<teacher>/videos.json -- see services/teacher_videos.py) to full
    VideoRef records, in the same order as video_ids. Ids not found in the
    catalog are silently skipped rather than raising, since a teacher's video
    list is hand-authored and shouldn't 500 the whole reply over one typo'd id."""
    if not video_ids:
        return []
    placeholders = ",".join("?" for _ in video_ids)
    with _connect() as con:
        rows = con.execute(
            f"SELECT video_id, title, description, url, thumbnail_url FROM videos WHERE video_id IN ({placeholders})",
            video_ids,
        ).fetchall()
    by_id = {r[0]: VideoRef(video_id=r[0], title=r[1], description=r[2], url=r[3], thumbnail_url=r[4] or "") for r in rows}
    return [by_id[vid] for vid in video_ids if vid in by_id]
