"""Build train_raw.jsonl and choose a stable reference clip (data-prep step 05).

Ported from ``scripts/05_make_dataset.py``. Reference scoring and duration
windows come from :class:`~voice_clone_lab.config.AudioConfig`; the formulas
are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean

import soundfile as sf

from . import config as _config
from .audio import audio_stats, load_audio, warning_lines
from .config import AudioConfig, VoicePaths, rel_to_root
from .utils import (
    read_jsonl,
    refuse_any_existing,
    refuse_overwrite,
    resolve_artifact_path,
    write_json,
    write_jsonl,
)

__all__ = ["build_dataset"]


def _dataset_path(path: Path, mode: str, qwen_finetuning_dir: Path) -> str:
    path = path.resolve()
    if mode == "absolute":
        return str(path)
    if mode == "project-relative":
        return rel_to_root(path)
    if mode == "qwen-relative":
        # Relative to the actual Qwen finetuning dir (old script hardcoded ../../..).
        return os.path.relpath(path, qwen_finetuning_dir)
    raise SystemExit(f"Unsupported path_mode: {mode}")


def _validate_reference(path: Path, sample_rate: int, cfg: AudioConfig) -> dict:
    audio = load_audio(path, sample_rate)
    stats = audio_stats(audio, sample_rate)
    if stats["duration_seconds"] < cfg.reference_min_seconds:
        raise SystemExit(
            f"Reference audio is too short ({stats['duration_seconds']:.1f}s). "
            f"Use at least {cfg.reference_min_seconds:.1f}s."
        )
    if stats["duration_seconds"] > 20:
        print(
            f"WARN: reference audio is long ({stats['duration_seconds']:.1f}s). "
            f"Prefer {cfg.reference_ideal_min_seconds:.0f}-{cfg.reference_ideal_max_seconds:.0f}s."
        )
    if not (cfg.reference_ideal_min_seconds <= stats["duration_seconds"] <= cfg.reference_ideal_max_seconds):
        print(
            f"WARN: reference is {stats['duration_seconds']:.1f}s. Ideal range is "
            f"{cfg.reference_ideal_min_seconds:.0f}-{cfg.reference_ideal_max_seconds:.0f}s."
        )
    for warning in warning_lines(stats):
        print(f"WARN reference: {warning}")
    return stats


def _write_reference(source: Path, dest: Path, sample_rate: int, force: bool) -> None:
    if dest.exists() and source.resolve() == dest.resolve():
        return
    refuse_overwrite(dest, force, "reference audio")
    audio = load_audio(source, sample_rate)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio, sample_rate, subtype="PCM_16")


def _choose_reference(valid_rows: list[dict], sample_rate: int, cfg: AudioConfig) -> Path:
    """Pick the cleanest in-window chunk: closest to the target duration, then
    penalized for RMS below -24 dBFS."""
    candidates: list[tuple[float, Path]] = []
    for row in valid_rows:
        path = Path(row["audio_path"])
        try:
            audio = load_audio(path, sample_rate)
        except Exception:
            continue
        stats = audio_stats(audio, sample_rate)
        duration = stats["duration_seconds"]
        if (
            cfg.reference_ideal_min_seconds <= duration <= cfg.reference_ideal_max_seconds
            and stats["estimated_snr_db"] >= 10
            and stats["clipped_samples"] == 0
        ):
            score = abs(duration - cfg.target_chunk_seconds) + max(0.0, -24.0 - stats["rms_dbfs"]) * 0.05
            candidates.append((score, path))
    if not candidates:
        raise SystemExit(
            f"Could not auto-select a {cfg.reference_ideal_min_seconds:.0f}-"
            f"{cfg.reference_ideal_max_seconds:.0f}s clean reference clip. "
            "Re-run with source_ref=path/to/ref.wav."
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def build_dataset(
    paths: VoicePaths,
    cfg: AudioConfig,
    sample_rate: int,
    min_minutes: float = 5.0,
    source_ref: Path | None = None,
    path_mode: str = "qwen-relative",
    qwen_finetuning_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Build ``paths.train_raw.jsonl`` + ``paths.dataset_stats`` (step 05).

    Rows are ``{"audio", "text", "ref_audio"}`` with paths rendered per
    ``path_mode`` (qwen-relative | project-relative | absolute). Returns stats.
    """
    if qwen_finetuning_dir is None:
        qwen_finetuning_dir = _config.ROOT / "third_party" / "Qwen3-TTS" / "finetuning"
    if not paths.transcripts_jsonl.exists():
        raise SystemExit(f"Transcripts file does not exist: {paths.transcripts_jsonl}")
    refuse_any_existing([paths.train_raw_jsonl, paths.dataset_stats], force, "dataset outputs")

    valid: list[dict] = []
    rejected: list[dict] = []
    durations: list[float] = []
    for row in read_jsonl(paths.transcripts_jsonl):
        audio_path = resolve_artifact_path(str(row.get("audio", "")).strip())
        text = str(row.get("text", "")).strip()
        if not audio_path.exists():
            rejected.append({"audio": str(row.get("audio", "")), "text": text, "reason": "audio file missing"})
            continue
        if not text:
            rejected.append({"audio": str(row.get("audio", "")), "text": text, "reason": "empty transcript"})
            continue
        try:
            audio = load_audio(audio_path, sample_rate)
            stats = audio_stats(audio, sample_rate)
        except Exception as exc:
            rejected.append({"audio": str(row.get("audio", "")), "text": text, "reason": f"failed to read audio: {exc}"})
            continue
        durations.append(stats["duration_seconds"])
        valid.append({"audio_path": str(audio_path), "text": text, "stats": stats})

    if not valid:
        raise SystemExit("No valid transcript rows found.")

    ref_audio = paths.reference_audio
    if source_ref is not None:
        _write_reference(Path(source_ref), ref_audio, sample_rate, force=force)
        print(f"Copied reference audio to: {ref_audio}")
    elif ref_audio.exists():
        print(f"Using existing reference audio: {ref_audio}")
    else:
        selected = _choose_reference(valid, sample_rate, cfg)
        _write_reference(selected, ref_audio, sample_rate, force=force)
        print(f"Auto-selected reference audio from: {selected}")
        print(f"Wrote reference audio to: {ref_audio}")

    ref_stats = _validate_reference(ref_audio, sample_rate, cfg)
    train_rows = [
        {
            "audio": _dataset_path(Path(row["audio_path"]), path_mode, qwen_finetuning_dir),
            "text": row["text"],
            "ref_audio": _dataset_path(ref_audio, path_mode, qwen_finetuning_dir),
        }
        for row in valid
    ]

    total_minutes = sum(durations) / 60.0
    stats_doc = {
        "clips": len(valid),
        "total_minutes": total_minutes,
        "min_duration_seconds": min(durations),
        "max_duration_seconds": max(durations),
        "mean_duration_seconds": mean(durations),
        "rejected_clips": len(rejected),
        "rejected": rejected,
        "reference_audio": str(ref_audio),
        "reference_stats": ref_stats,
        "path_mode": path_mode,
    }

    write_jsonl(paths.train_raw_jsonl, train_rows, force=force)
    write_json(paths.dataset_stats, stats_doc, force=force)

    print(f"Wrote dataset: {paths.train_raw_jsonl}")
    print(f"Wrote dataset stats: {paths.dataset_stats}")
    print(f"Clips: {len(valid)}")
    print(f"Total minutes: {total_minutes:.2f}")
    print(
        f"Duration min/max/mean: {min(durations):.2f}s / "
        f"{max(durations):.2f}s / {mean(durations):.2f}s"
    )
    print(f"Rejected clips: {len(rejected)}")
    if total_minutes < min_minutes:
        print(
            f"WARN: only {total_minutes:.2f} minutes of valid data. "
            f"Recommended minimum is {min_minutes:.1f} minutes."
        )
    return stats_doc
