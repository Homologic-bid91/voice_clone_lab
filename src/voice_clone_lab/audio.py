"""Audio extraction, cleaning, and chunking (data-prep steps 01-03).

Ported from ``scripts/01_extract_audio.sh``, ``scripts/02_clean_audio.py``,
``scripts/03_split_audio.py``, and the audio helpers in ``scripts/common.py``.
Filter chains, thresholds, and scoring are unchanged; threshold values now come
from :class:`~voice_clone_lab.config.AudioConfig`.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import AudioConfig, VoicePaths, rel_to_root
from .utils import (
    command_exists,
    ensure_parent,
    refuse_any_existing,
    refuse_overwrite,
    write_json,
)

__all__ = [
    "load_audio",
    "audio_stats",
    "warning_lines",
    "extract_audio",
    "clean_audio",
    "split_audio",
]

FFMPEG_FILTER_CHAIN = "loudnorm=I=-20:TP=-1.5:LRA=11,alimiter=limit=0.95"


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 resampled to ``sample_rate``."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def audio_stats(samples: np.ndarray, sample_rate: int) -> dict:
    """Quality metrics for a mono signal: levels, clipping, and estimated SNR."""
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = samples.astype(np.float32, copy=False)
    duration = float(len(samples) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    rms_dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    clipped = int(np.sum(np.abs(samples) >= 0.999))

    # Frame RMS at 30 ms frames / 10 ms hop, vectorized with a sliding window.
    # Same framing as the legacy loop: full frames at 0, hop, 2*hop, ... up to
    # len-frame; a single partial frame when shorter than one frame.
    frame = max(1, int(sample_rate * 0.03))
    hop = max(1, int(sample_rate * 0.01))
    n = len(samples)
    if n >= frame:
        starts = np.arange(0, n - frame + 1, hop)
        frames = np.lib.stride_tricks.sliding_window_view(samples, frame)[starts]
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    elif n:
        frame_rms = np.array([np.sqrt(np.mean(np.square(samples)))])
    else:
        frame_rms = np.empty(0)

    if frame_rms.size:
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
        noise_floor_dbfs = float(np.percentile(frame_db, 10))
        speech_floor_dbfs = float(np.percentile(frame_db, 90))
        estimated_snr_db = float(speech_floor_dbfs - noise_floor_dbfs)
    else:
        noise_floor_dbfs = -120.0
        speech_floor_dbfs = -120.0
        estimated_snr_db = 0.0

    return {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "peak": peak,
        "rms_dbfs": rms_dbfs,
        "clipped_samples": clipped,
        "noise_floor_dbfs": noise_floor_dbfs,
        "speech_floor_dbfs": speech_floor_dbfs,
        "estimated_snr_db": estimated_snr_db,
    }


def warning_lines(stats: dict) -> list[str]:
    """Human-readable warnings for problematic audio stats."""
    warnings: list[str] = []
    if stats["clipped_samples"] > 0:
        warnings.append(f"clipping detected: {stats['clipped_samples']} samples at or above 0.999")
    if stats["rms_dbfs"] < -35:
        warnings.append(f"audio is very quiet: RMS {stats['rms_dbfs']:.1f} dBFS")
    if stats["estimated_snr_db"] < 12:
        warnings.append(f"audio may be noisy: estimated SNR {stats['estimated_snr_db']:.1f} dB")
    if stats["peak"] > 0.98:
        warnings.append(f"peak level is high: {stats['peak']:.3f}")
    return warnings


def extract_audio(input_path: Path, output_path: Path, sample_rate: int, force: bool = False) -> None:
    """Extract the first audio stream to a mono WAV via ffmpeg (step 01)."""
    if not command_exists("ffmpeg"):
        raise SystemExit("ffmpeg is required. Install with: sudo apt-get install ffmpeg")
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    refuse_overwrite(output_path, force, "Output")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Input and output must be different files. Raw files are never modified in place.")
    ensure_parent(output_path)

    cmd = [
        "ffmpeg",
        "-y" if force else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        FFMPEG_FILTER_CHAIN,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed with exit code {result.returncode}:\n{result.stderr.strip()}")
    print(f"Wrote mono {sample_rate} Hz WAV: {output_path}")


def _trim_long_silence(audio: np.ndarray, sr: int, top_db: float, padding_ms: int, join_silence_ms: int) -> np.ndarray:
    """Remove long silences, keeping ``padding_ms`` around speech and joining
    remaining gaps with at most ``join_silence_ms`` of silence."""
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=256)
    if len(intervals) == 0:
        return audio

    pad = int(sr * padding_ms / 1000)
    max_join = int(sr * join_silence_ms / 1000)
    padded: list[tuple[int, int]] = []
    for start, end in intervals:
        padded.append((max(0, int(start) - pad), min(len(audio), int(end) + pad)))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(padded):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[np.ndarray] = []
    last_end = 0
    for start, end in merged:
        if parts and start > last_end:
            gap = start - last_end
            parts.append(np.zeros(min(gap, max_join), dtype=np.float32))
        parts.append(audio[start:end])
        last_end = end
    return np.concatenate(parts) if parts else audio


def _maybe_denoise(audio: np.ndarray, sr: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return audio
    try:
        import noisereduce as nr
    except Exception:
        print("WARN: denoise requested but noisereduce is not installed. Continuing without denoise.")
        return audio
    return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.35).astype(np.float32)


def clean_audio(
    input_path: Path,
    output_path: Path,
    cfg: AudioConfig,
    sample_rate: int,
    denoise: bool = False,
    force: bool = False,
) -> dict:
    """Conservatively clean extracted audio (step 02): trim long silences,
    optionally denoise, and rescale hot peaks. Returns before/after stats."""
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Input and output must be different files. Raw/extracted files are not modified in place.")
    refuse_overwrite(output_path, force, "cleaned audio")

    audio = load_audio(input_path, sample_rate)
    before = audio_stats(audio, sample_rate)
    cleaned = _maybe_denoise(audio, sample_rate, denoise)
    cleaned = _trim_long_silence(
        cleaned,
        sample_rate,
        top_db=cfg.silence_top_db,
        padding_ms=cfg.clean_padding_ms,
        join_silence_ms=cfg.clean_join_silence_ms,
    )

    peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
    if peak > 0.98:
        cleaned = cleaned * (0.95 / peak)

    after = audio_stats(cleaned, sample_rate)
    ensure_parent(output_path)
    sf.write(output_path, cleaned, sample_rate, subtype="PCM_16")

    warnings = [f"input: {w}" for w in warning_lines(before)]
    warnings += [f"output: {w}" for w in warning_lines(after)]
    print(f"Input duration:  {before['duration_seconds'] / 60:.2f} min")
    print(f"Output duration: {after['duration_seconds'] / 60:.2f} min")
    for warning in warnings:
        print(f"WARN {warning}")
    print(f"Wrote cleaned audio: {output_path}")
    return {"before": before, "after": after, "warnings": warnings}


def _webrtcvad_intervals(audio: np.ndarray, sr: int, aggressiveness: int) -> np.ndarray | None:
    """Speech intervals via WebRTC VAD, or None when unavailable/failed."""
    try:
        import webrtcvad
    except Exception:
        return None

    target_sr = 16000
    try:
        y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr) if sr != target_sr else audio
        y = np.clip(y, -1.0, 1.0)
        pcm = (y * 32767).astype(np.int16)
        vad = webrtcvad.Vad(aggressiveness)
        frame_ms = 30
        frame_len = int(target_sr * frame_ms / 1000)
        speech_frames: list[tuple[int, int]] = []
        for start in range(0, len(pcm) - frame_len + 1, frame_len):
            frame = pcm[start : start + frame_len].tobytes()
            if vad.is_speech(frame, target_sr):
                speech_frames.append((start, start + frame_len))
    except Exception as exc:
        print(f"WARN: WebRTC VAD failed; falling back to silence detection: {exc}")
        return None

    if not speech_frames:
        return None

    merged: list[tuple[int, int]] = []
    gap = int(0.35 * target_sr)
    cur_start, cur_end = speech_frames[0]
    for start, end in speech_frames[1:]:
        if start - cur_end <= gap:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    scale = sr / target_sr
    min_len = int(0.25 * sr)
    return np.array(
        [[int(start * scale), int(end * scale)] for start, end in merged if int((end - start) * scale) >= min_len],
        dtype=np.int64,
    )


def _speech_intervals(audio: np.ndarray, sr: int, top_db: float, aggressiveness: int) -> np.ndarray:
    vad = _webrtcvad_intervals(audio, sr, aggressiveness)
    if vad is not None and len(vad) > 0:
        print("Using WebRTC VAD speech detection.")
        return vad

    print("Using librosa silence-based speech detection.")
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=256)
    min_len = int(0.25 * sr)
    return np.array([[s, e] for s, e in intervals if e - s >= min_len], dtype=np.int64)


def _build_segments(
    intervals: np.ndarray, sr: int, min_sec: float, max_sec: float, pad_sec: float
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Merge padded speech intervals into chunks bounded by min/max seconds."""
    segments: list[tuple[int, int]] = []
    rejected: list[dict] = []
    if len(intervals) == 0:
        return segments, [{"reason": "no speech intervals detected"}]

    max_len = int(max_sec * sr)
    min_len = int(min_sec * sr)
    pad = int(pad_sec * sr)

    current_start: int | None = None
    current_end: int | None = None

    for raw_start, raw_end in intervals:
        start = max(0, int(raw_start) - pad)
        end = int(raw_end) + pad
        if end - start > max_len:
            rejected.append({
                "reason": "speech run longer than max chunk duration",
                "start_seconds": start / sr,
                "end_seconds": end / sr,
                "duration_seconds": (end - start) / sr,
            })
            if current_start is not None and current_end is not None and current_end - current_start >= min_len:
                segments.append((current_start, current_end))
            current_start = None
            current_end = None
            continue

        if current_start is None:
            current_start, current_end = start, end
            continue

        assert current_end is not None
        candidate_end = end
        if candidate_end - current_start <= max_len:
            current_end = candidate_end
            continue

        if current_end - current_start >= min_len:
            segments.append((current_start, current_end))
        else:
            rejected.append({
                "reason": "candidate chunk too short",
                "start_seconds": current_start / sr,
                "end_seconds": current_end / sr,
                "duration_seconds": (current_end - current_start) / sr,
            })
        current_start, current_end = start, end

    if current_start is not None and current_end is not None:
        if current_end - current_start >= min_len:
            segments.append((current_start, current_end))
        else:
            rejected.append({
                "reason": "final chunk too short",
                "start_seconds": current_start / sr,
                "end_seconds": current_end / sr,
                "duration_seconds": (current_end - current_start) / sr,
            })

    return segments, rejected


def _reject_reason(stats: dict, cfg: AudioConfig) -> str | None:
    duration = stats["duration_seconds"]
    if duration < cfg.min_chunk_seconds:
        return "too short"
    if duration > cfg.max_chunk_seconds:
        return "too long"
    if stats["rms_dbfs"] < cfg.min_rms_dbfs:
        return "silent or too quiet"
    if stats["clipped_samples"] > max(10, int(0.0005 * stats["sample_rate"] * duration)):
        return "clipped"
    if stats["estimated_snr_db"] < cfg.min_snr_db:
        return "mostly noise or weak speech separation"
    return None


def split_audio(input_path: Path, paths: VoicePaths, cfg: AudioConfig, sample_rate: int, force: bool = False) -> dict:
    """Split cleaned speech into fine-tuning chunks (step 03).

    Writes ``chunk_0001.wav``... into ``paths.chunks_dir`` and metadata (with
    project-relative clip paths) to ``paths.chunk_metadata``. Returns metadata.
    """
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    chunks_dir = paths.chunks_dir
    metadata_path = paths.chunk_metadata
    existing_outputs = list(chunks_dir.glob("chunk_*.wav")) + [metadata_path]
    refuse_any_existing(existing_outputs, force, "chunk outputs")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in chunks_dir.glob("chunk_*.wav"):
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    audio = load_audio(input_path, sample_rate)
    intervals = _speech_intervals(audio, sample_rate, cfg.silence_top_db, cfg.vad_aggressiveness)
    segments, rejected = _build_segments(
        intervals,
        sample_rate,
        min_sec=cfg.min_chunk_seconds,
        max_sec=cfg.max_chunk_seconds,
        pad_sec=cfg.pad_seconds,
    )

    clips: list[dict] = []
    for idx, (start, end) in enumerate(segments, start=1):
        chunk = audio[start : min(end, len(audio))]
        stats = audio_stats(chunk, sample_rate)
        reason = _reject_reason(stats, cfg)
        if reason:
            rejected.append({
                "reason": reason,
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
                **stats,
            })
            continue
        path = chunks_dir / f"chunk_{idx:04d}.wav"
        sf.write(path, chunk, sample_rate, subtype="PCM_16")
        clips.append({
            "path": rel_to_root(path),
            "start_seconds": start / sample_rate,
            "end_seconds": end / sample_rate,
            **stats,
        })

    metadata = {
        "source": rel_to_root(input_path),
        "sample_rate": sample_rate,
        "settings": {
            "min_sec": cfg.min_chunk_seconds,
            "max_sec": cfg.max_chunk_seconds,
            "top_db": cfg.silence_top_db,
        },
        "clips": clips,
        "rejected": rejected,
    }
    write_json(metadata_path, metadata, force=force)

    total_minutes = sum(c["duration_seconds"] for c in clips) / 60.0
    print(f"Wrote {len(clips)} chunks to {chunks_dir}")
    print(f"Accepted duration: {total_minutes:.2f} min")
    print(f"Rejected chunks/regions: {len(rejected)}")
    for warning in warning_lines(audio_stats(audio, sample_rate)):
        print(f"WARN source: {warning}")
    print(f"Wrote metadata: {metadata_path}")
    return metadata
