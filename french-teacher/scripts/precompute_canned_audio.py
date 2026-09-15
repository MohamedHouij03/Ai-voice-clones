"""
Warms the on-disk canned-audio cache (voices/<teacher>/canned_audio/*.wav) for
every configured teacher, so even the *first* real session is fast instead of
only the second one onward.

This is optional maintenance tooling, not a hard requirement: conversation_service
populates the same cache lazily the first time a line is actually needed in a
real session. Run this after adding a new teacher, or after editing
backend/services/canned_responses.py or backend/data/common_phrases.json, to
pre-pay that cost up front instead of on the next real conversation.

Usage:
    python scripts/precompute_canned_audio.py [--teacher teacher_3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.services.canned_responses import all_entries_for_precompute
from backend.services.chatterbox_service import get_chatterbox_service
from backend.services.common_phrases import all_entries_for_precompute as common_phrase_entries
from backend.services.conversation_service import (
    _cached_audio_path,
    _split_sentences,
    _synthesize,
    build_greeting_text,
)
from backend.services.topics import topic_menu_text

import soundfile as sf  # noqa: F401  (imported by conversation_service; keep env explicit)


def _display_name(teacher_dir: Path) -> str:
    meta_path = teacher_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8")).get("display_name", teacher_dir.name)
    return teacher_dir.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", help="Only warm this one teacher folder (e.g. teacher_3). Default: all.")
    args = p.parse_args()

    if not config.VOICES_DIR.exists():
        print(f"[ERROR] {config.VOICES_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    teacher_dirs = sorted(
        d for d in config.VOICES_DIR.iterdir()
        if d.is_dir() and (d / "reference.wav").exists() and (not args.teacher or d.name == args.teacher)
    )
    if not teacher_dirs:
        print("[ERROR] No matching teacher folder with a reference.wav found.", file=sys.stderr)
        sys.exit(1)

    chatterbox = get_chatterbox_service()

    for teacher_dir in teacher_dirs:
        teacher_voice = teacher_dir.name
        display_name = _display_name(teacher_dir)
        reference_path = teacher_dir / "reference.wav"

        topic_menu = topic_menu_text(teacher_voice)
        greeting_text = build_greeting_text(display_name, topic_menu)
        # process_message_stream's canned-reply branch always splits canned["text"]
        # into sentences and caches each under f"{key}_{i}" (see conversation_service.py),
        # even when there's only one sentence -- so precompute must mirror that
        # exact split+index scheme, not cache the raw unsplit text under "key" alone
        # (every canned_responses entry happens to be 2 sentences today, so caching
        # the bare key was previously 100% dead weight -- never read at runtime).
        canned_lines = []
        for entry in all_entries_for_precompute(display_name, topic_menu):
            for i, sentence in enumerate(_split_sentences(entry["text"])):
                canned_lines.append({"key": f"{entry['key']}_{i}", "text": sentence})

        lines = [{"key": "greeting", "text": greeting_text}] + canned_lines + common_phrase_entries()

        print(f"\n=== {teacher_voice} ({display_name}) ===")
        for line in lines:
            cache_path = _cached_audio_path(teacher_voice, line["key"])
            if cache_path.exists():
                print(f"[SKIP] {line['key']} already cached at {cache_path}")
                continue
            t0 = time.perf_counter()
            _synthesize(chatterbox, teacher_voice, reference_path, line["text"], cache_key=line["key"])
            print(f"[OK]   {line['key']} ({time.perf_counter() - t0:.1f}s) -> {cache_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
