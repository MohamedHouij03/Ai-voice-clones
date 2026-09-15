"""
Phase 1 proof of concept: Chatterbox Multilingual V3 voice cloning for French.

Loads a teacher reference WAV, synthesizes French speech in that voice using
ChatterboxMultilingualTTS (t3_model="v3"), and saves the result to disk.

This script intentionally has no web framework, no session logic, and no LLM/STT.
Its only job is to prove that voice cloning works end-to-end on this machine.

Usage:
    python scripts/phase1_poc.py --text "Bonjour, comment allez-vous ?" --reference ../voices/teacher_1/reference.wav
"""

import argparse
import sys
import time
from pathlib import Path


def estimate_speech_rate(y, sr):
    """
    Rough syllable-rate estimate (syllables/sec) from the audio's energy envelope,
    via peak-picking on smoothed RMS (an approximation of syllable-nuclei counting).
    Not phonetically precise, but consistent enough between two clips to compute a
    pace ratio between them.
    """
    import numpy as np
    import librosa
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    hop = int(0.010 * sr)
    frame = int(0.025 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_smooth = uniform_filter1d(rms, size=5)
    rms_db = librosa.amplitude_to_db(rms_smooth, ref=np.max)
    voiced_duration = (rms_db > -35).sum() * hop / sr
    if voiced_duration <= 0:
        return 0.0
    min_dist = max(1, int(0.12 * sr / hop))
    peaks, _ = find_peaks(rms_smooth, distance=min_dist, prominence=np.std(rms_smooth) * 0.3)
    return len(peaks) / voiced_duration


def wsola_time_stretch(y, rate, sr, frame_ms=30, tolerance_ms=15):
    """
    Time-domain (WSOLA) time-stretch: pitch-preserving like librosa's phase-vocoder
    time_stretch(), but works by re-splicing raw waveform segments at their best
    cross-correlation alignment instead of reconstructing STFT phase. This avoids
    the smeared-harmonics/"reverb" artifact that phase-vocoder stretching produces
    at larger stretch ratios, which is what a synthesis_hop-sized time-stretch of
    generated TTS speech turned out to sound like here.

    rate: same convention as librosa.effects.time_stretch (rate<1 slows down).
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    y = np.asarray(y, dtype=np.float32)
    frame_length = int(frame_ms / 1000 * sr)
    synthesis_hop = frame_length // 2  # 50% overlap: Hann window satisfies constant-overlap-add
    tolerance = int(tolerance_ms / 1000 * sr)
    analysis_hop = synthesis_hop * rate
    overlap_len = frame_length - synthesis_hop
    window = np.hanning(frame_length).astype(np.float32)

    n_in = len(y)
    est_out_len = int(n_in / rate) + frame_length + 1
    output = np.zeros(est_out_len, dtype=np.float32)
    window_sum = np.zeros(est_out_len, dtype=np.float32)

    input_pos = 0.0
    output_pos = 0
    prev_raw_tail = None

    while True:
        ideal_pos = int(round(input_pos))
        if ideal_pos + frame_length > n_in:
            break

        if prev_raw_tail is not None:
            lo = max(0, ideal_pos - tolerance)
            hi = min(n_in - frame_length, ideal_pos + tolerance)
            if hi <= lo:
                pos = ideal_pos
            else:
                candidates = sliding_window_view(y[lo:hi + overlap_len], overlap_len)
                scores = candidates @ prev_raw_tail
                pos = lo + int(np.argmax(scores))
        else:
            pos = ideal_pos

        segment = y[pos:pos + frame_length]
        output[output_pos:output_pos + frame_length] += segment * window
        window_sum[output_pos:output_pos + frame_length] += window
        prev_raw_tail = segment[synthesis_hop:]  # raw (unwindowed) tail, for next splice-point search

        output_pos += synthesis_hop
        input_pos += analysis_hop

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    return output[:output_pos + frame_length]


def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox Multilingual V3 French voice-cloning POC")
    parser.add_argument(
        "--text",
        default="Bonjour, comment allez-vous aujourd'hui ? J'espère que vous passez une bonne journée.",
        help="French text to synthesize.",
    )
    parser.add_argument(
        "--reference",
        default=str(Path(__file__).resolve().parent.parent / "voices" / "teacher_1" / "reference.wav"),
        help="Path to the teacher reference WAV used for voice cloning.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "output" / "phase1_output.wav"),
        help="Where to save the generated WAV.",
    )
    parser.add_argument("--language", default="fr", help="Chatterbox language_id.")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Chatterbox exaggeration parameter.")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Chatterbox cfg_weight parameter.")
    parser.add_argument("--t3-model", default="v3", help="Chatterbox multilingual T3 checkpoint ('v3' or 'v2').")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Manual pace override applied after generation (1.0=unchanged, 0.7=70%% speed/slower, "
             "pitch-preserving). If omitted, pace is auto-matched to the reference recording's "
             "estimated speaking rate (see --no-pace-match to disable).",
    )
    parser.add_argument(
        "--no-pace-match",
        action="store_true",
        help="Disable automatic pace matching against the reference recording; use the model's raw output pace.",
    )
    parser.add_argument(
        "--pace-strength",
        type=float,
        default=0.0,
        help="How much of the full reference-pace correction to apply, 0.0-1.0 (default 0.0 = off). "
             "Any amount of time-stretching (even the WSOLA method used here) trades some audio "
             "fidelity for pace -- default is off so plain generation matches the model's raw output "
             "quality. 1.0 = fully match the reference's estimated speaking rate. Ignored if --speed is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    reference_path = Path(args.reference)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Error handling: missing reference voice (section 21) ---
    if not reference_path.exists():
        print(f"[ERROR] Teacher reference audio not found: {reference_path}", file=sys.stderr)
        print(
            "        Place a clean, single-speaker WAV recording of the authorized teacher's "
            "voice at that path (see README.md, section 'Voice reference'). This must be an "
            "authorized recording of a real teacher (see consent policy in the README).",
            file=sys.stderr,
        )
        sys.exit(1)

    t_import_start = time.perf_counter()
    try:
        import torch
        import soundfile as sf
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
        print("        Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
    t_import = time.perf_counter() - t_import_start

    # --- Device selection: CUDA with CPU fallback (section 18) ---
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] CUDA available -> using GPU: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("[WARN] CUDA not available -> falling back to CPU. Generation will be much slower.")

    print(f"[INFO] Loading ChatterboxMultilingualTTS (t3_model={args.t3_model}) on device='{device}'...")
    t_load_start = time.perf_counter()
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=args.t3_model)
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA out-of-memory while loading the model.", file=sys.stderr)
        print("        Close other GPU-using applications, or set device to 'cpu' as a fallback.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load Chatterbox model: {e}", file=sys.stderr)
        sys.exit(1)
    t_load = time.perf_counter() - t_load_start
    print(f"[INFO] Model loaded in {t_load:.2f}s")

    print(f"[INFO] Reference voice : {reference_path}")
    print(f"[INFO] Language        : {args.language}")
    print(f"[INFO] Text            : {args.text}")

    t_gen_start = time.perf_counter()
    try:
        wav = model.generate(
            args.text,
            language_id=args.language,
            audio_prompt_path=str(reference_path),
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA out-of-memory during generation. Try shorter text.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    t_gen = time.perf_counter() - t_gen_start

    # Note: torchaudio.save() in this torchaudio version requires the separate
    # `torchcodec` package, which is not part of this project's dependency set.
    # soundfile (already installed transitively via librosa) writes WAV directly
    # from the numpy array without that extra dependency.
    wav_np = wav.squeeze(0).detach().cpu().numpy()

    # --- Pace matching ---
    # Chatterbox's generate() has no speaking-rate parameter, and by default tends
    # to speak noticeably faster than the reference recording. Rather than guessing,
    # estimate both clips' speech rate and time-stretch (pitch-preserving) to match.
    if args.speed is not None:
        applied_rate = args.speed
        print(f"[INFO] Applying manual pace override: rate={applied_rate:.3f}")
    elif not args.no_pace_match:
        import librosa
        ref_y, _ = librosa.load(str(reference_path), sr=model.sr, mono=True)
        ref_rate = estimate_speech_rate(ref_y, model.sr)
        gen_rate = estimate_speech_rate(wav_np, model.sr)
        if ref_rate > 0 and gen_rate > 0:
            full_correction = max(0.5, min(1.0, ref_rate / gen_rate))
        else:
            full_correction = 1.0
        strength = max(0.0, min(1.0, args.pace_strength))
        # Blend between no correction (1.0) and the full reference-match ratio, rather
        # than always fully matching it -- full correction measured as too slow/robotic.
        applied_rate = 1.0 - strength * (1.0 - full_correction)
        print(
            f"[INFO] Pace match: reference~{ref_rate:.2f} syl/s, generated~{gen_rate:.2f} syl/s, "
            f"full correction would be rate={full_correction:.3f} -> applying {strength:.0%} of it: "
            f"rate={applied_rate:.3f}"
        )
    else:
        applied_rate = 1.0

    if applied_rate != 1.0:
        wav_np = wsola_time_stretch(wav_np, rate=applied_rate, sr=model.sr)

    sf.write(str(output_path), wav_np, model.sr, subtype="PCM_16")

    t_total = t_load + t_gen
    print()
    print("---- Timing ----")
    print(f"Import   : {t_import:.2f}s")
    print(f"Model load: {t_load:.2f}s (one-time cost; a running server keeps this in memory)")
    print(f"Generate : {t_gen:.2f}s")
    print(f"Total (load+generate): {t_total:.2f}s")
    print("----------------")
    print(f"[OK] Saved generated speech to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
