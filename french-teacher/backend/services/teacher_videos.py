"""
Per-teacher video recommendations, shown once a topic quiz finishes with that
teacher (see topics.advance_topic_quiz). These are real videos from
Dreaming's own YouTube catalog (video_library.py) that happen to feature this
particular teacher -- voices/<teacher>/videos.json only needs to record WHICH
catalog video ids belong to them; title/description/thumbnail are resolved
from the existing catalog, not duplicated here.

Distinct from video_library.find_relevant_video(), which is keyword-matched
against whatever topic came up in open conversation and fires only when a
student explicitly asks for "une vidéo" -- this fires automatically once a
topic quiz completes, and is scoped to that one teacher's own videos.
"""
import json
import logging

from .. import config
from .video_library import VideoRef, get_videos_by_ids

logger = logging.getLogger(__name__)

# videos.json is authored offline, same caching assumption as topics.py's
# own _topics_cache for topics.json.
_video_ids_cache: dict = {}


def _load_teacher_video_ids(teacher_voice: str) -> list:
    if teacher_voice in _video_ids_cache:
        return _video_ids_cache[teacher_voice]

    videos_path = config.VOICES_DIR / teacher_voice / "videos.json"
    ids = []
    if videos_path.exists():
        try:
            ids = json.loads(videos_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse videos.json for %s", teacher_voice, exc_info=True)
            ids = []
    _video_ids_cache[teacher_voice] = ids
    return ids


def teacher_video_recommendations(teacher_voice: str) -> list:
    """This teacher's own catalog videos, as full VideoRef records -- or []
    if this teacher has no videos.json configured yet."""
    return get_videos_by_ids(_load_teacher_video_ids(teacher_voice))
