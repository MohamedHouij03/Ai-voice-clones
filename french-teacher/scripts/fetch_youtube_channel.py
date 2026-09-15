"""
Fetches a YouTube channel's video catalog via YouTube's official Data API v3
(channels.list to find the channel's uploads playlist, then paginated
playlistItems.list over it), storing title/description/url in the local
video-library SQLite DB (see backend/services/video_library.py) so the
conversation teacher can occasionally reference a real video for a topic
that came up.

Deliberately NOT scraping Dreaming's own app (app.dreaming.com) for this:
its robots.txt explicitly disallows ClaudeBot -- and every other major AI
company's crawler -- from crawling it at all, alongside a more permissive
general policy for search engines. That's a clear, deliberate signal not to
build an AI-driven scraper against that site, regardless of user-agent.
YouTube's official Data API is a publicly documented, ToS-sanctioned way to
get that same channel's public video metadata instead. Default channel is
Dreaming French's own: https://www.youtube.com/channel/UCG7lancLEOKXZ7lEEyzvwjA

Requires a free YouTube Data API v3 key: Google Cloud Console -> select/create
a project -> "APIs & Services" -> Library -> enable "YouTube Data API v3" ->
Credentials -> Create API key. Put it in .env as YOUTUBE_API_KEY. The free
quota (10,000 units/day by default) is far more than one channel sync needs
(a channels.list call plus one playlistItems.list call per 50 videos).

Usage:
    python scripts/fetch_youtube_channel.py [--channel-id UCxxxxxxxx]
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.services.video_library import count_videos, upsert_video

API_BASE = "https://www.googleapis.com/youtube/v3"

# Real videos never have these exact titles -- YouTube substitutes them for
# entries that were deleted or made private after being added to the playlist.
_SKIP_TITLES = {"Deleted video", "Private video"}


def _get_uploads_playlist(client: httpx.Client, channel_id: str):
    resp = client.get(
        f"{API_BASE}/channels",
        params={"part": "contentDetails,snippet", "id": channel_id, "key": config.YOUTUBE_API_KEY},
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        print(f"[ERROR] No channel found for id={channel_id}. Check the channel ID and API key.", file=sys.stderr)
        sys.exit(1)
    channel_title = items[0]["snippet"]["title"]
    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return uploads_playlist_id, channel_title


def _iter_playlist_items(client: httpx.Client, playlist_id: str):
    page_token = None
    while True:
        params = {"part": "snippet", "playlistId": playlist_id, "maxResults": 50, "key": config.YOUTUBE_API_KEY}
        if page_token:
            params["pageToken"] = page_token
        resp = client.get(f"{API_BASE}/playlistItems", params=params)
        resp.raise_for_status()
        data = resp.json()
        yield from data.get("items", [])
        page_token = data.get("nextPageToken")
        if not page_token:
            return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel-id", default=config.DREAMING_YOUTUBE_CHANNEL_ID, help="YouTube channel ID (starts with UC).")
    args = p.parse_args()

    if not config.YOUTUBE_API_KEY:
        print("[ERROR] YOUTUBE_API_KEY is not set in .env. See this script's module docstring for how to get one.", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(timeout=30) as client:
        try:
            uploads_playlist_id, channel_title = _get_uploads_playlist(client, args.channel_id)
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] YouTube API request failed: {e}", file=sys.stderr)
            print(f"        Response: {e.response.text[:500]}", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Channel: {channel_title} (uploads playlist: {uploads_playlist_id})")

        synced = 0
        skipped = 0
        try:
            for item in _iter_playlist_items(client, uploads_playlist_id):
                snippet = item["snippet"]
                title = snippet["title"]
                if title in _SKIP_TITLES:
                    skipped += 1
                    continue

                video_id = snippet["resourceId"]["videoId"]
                thumbnails = snippet.get("thumbnails", {})
                thumbnail_url = (thumbnails.get("medium") or thumbnails.get("default") or {}).get("url")

                upsert_video(
                    video_id=video_id,
                    title=title,
                    description=snippet.get("description", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    published_at=snippet.get("publishedAt"),
                    thumbnail_url=thumbnail_url,
                    channel_title=channel_title,
                )
                synced += 1
                print(f"[OK] {title}")
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] YouTube API request failed mid-sync: {e}", file=sys.stderr)
            print(f"        Response: {e.response.text[:500]}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. {synced} videos synced ({skipped} deleted/private skipped). Total in DB: {count_videos()}")


if __name__ == "__main__":
    main()
