"""
Reference-audio preprocessing for Chatterbox voice cloning.

Chatterbox internally uses only the FIRST ~10 seconds of the reference clip for
timbre conditioning (`DEC_COND_LEN = 10 * S3GEN_SR`) and the first ~6 seconds for
prosody conditioning (`ENC_COND_LEN = 6 * S3_SR`) -- see
chatterbox/mtl_tts.py:prepare_conditionals(). Only the full-length clip is used
for the coarse speaker-identity embedding. In practice this means: whatever is
weakest in the first 10 seconds of your raw recording (silence, breath, a trailing
pause) directly hurts clone quality, even if the rest of the recording is great.

This script does NOT change Chatterbox's model or code. It only prepares a better
INPUT for it:
  1. Loudness-normalizes the clip to a consistent target (pyloudnorm).
  2. Trims leading/trailing silence.
  3. Slides a window across the clip to find the densest ~12s of continuous,
     consistently-voiced speech (least silence/dropout), instead of blindly
     assuming the first 12 seconds of the raw file are the best ones.
  4. Saves the result as mono/24kHz WAV (matching S3GEN_SR), ready to use as
     `audio_prompt_path`.

Usage:
    python scripts/prepare_reference.py --input voices/teacher_2/raw.wav --output voices/teacher_2/reference.wav
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

TARGET_SR = 24000  # matches Chatterbox's S3GEN_SR
WINDOW_SECONDS = 12.0  # covers DEC_COND_LEN (10s) with a little margin
TARGET_LUFS = -20.0


def parse_args():
    p = argparse.ArgumentParser(description="Clean and select the best segment of a teacher reference recording.")
    p.add_argument("--input", required=True, help="Path to the raw reference recording.")
    p.add_argument("--output", required=True, help="Where to save the cleaned reference WAV.")
    p.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    p.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    return p.parse_args()


def find_best_window(y: np.ndarray, sr: int, window_seconds: float) -> np.ndarray:
    window_len = int(window_seconds * sr)
    if len(y) <= window_len:
        return y

    frame = int(0.05 * sr)  # 50ms frames for energy profiling
    n_frames = len(y) // frame
    energies = np.array([
        np.sqrt(np.mean(y[i * frame:(i + 1) * frame] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    db = 20 * np.log10(energies + 1e-9)

    frames_per_window = max(1, window_len // frame)
    best_start_frame = 0
    best_score = -np.inf
    # Score a window by: high mean energy, and low silence-fraction (frames below -40 dBFS).
    for start in range(0, max(1, n_frames - frames_per_window + 1)):
        seg_db = db[start:start + frames_per_window]
        mean_db = seg_db.mean()
        silence_frac = np.mean(seg_db < -40)
        score = mean_db - 15.0 * silence_frac
        if score > best_score:
            best_score = score
            best_start_frame = start

    start_sample = best_start_frame * frame
    end_sample = min(start_sample + window_len, len(y))
    return y[start_sample:end_sample]


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading {in_path} ...")
    y, sr = librosa.load(str(in_path), sr=TARGET_SR, mono=True)
    raw_duration = len(y) / sr
    print(f"[INFO] Raw duration: {raw_duration:.2f}s at {sr} Hz (resampled/downmixed to mono)")

    # 1. Trim leading/trailing silence (top_db tuned for speech, not music).
    y_trimmed, trim_idx = librosa.effects.trim(y, top_db=35)
    trimmed_amount = (len(y) - len(y_trimmed)) / sr
    print(f"[INFO] Trimmed {trimmed_amount:.2f}s of leading/trailing silence")

    # 2. Loudness-normalize to a consistent target.
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_trimmed.astype(np.float64))
    y_norm = pyln.normalize.loudness(y_trimmed.astype(np.float64), loudness, args.target_lufs)
    peak = np.max(np.abs(y_norm))
    if peak > 0.99:
        y_norm = y_norm / peak * 0.97  # avoid clipping after normalization
    print(f"[INFO] Loudness normalized: {loudness:.1f} LUFS -> {args.target_lufs:.1f} LUFS")

    # 3. Pick the densest continuous window actually used for cloning.
    y_best = find_best_window(y_norm.astype(np.float32), sr, args.window_seconds)
    print(f"[INFO] Selected {len(y_best)/sr:.2f}s window with the most continuous, consistent speech")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_best, sr, subtype="PCM_16")
    print(f"[OK] Saved cleaned reference to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
