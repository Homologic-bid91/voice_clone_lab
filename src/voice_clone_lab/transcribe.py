"""Chunk transcription and review-file round-trip (data-prep step 04).

Ported from ``scripts/04_transcribe.py``. ASR backends are optional heavy
dependencies and are imported lazily; ``backend="auto"`` tries whisperx, then
faster-whisper, then whisper.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from .config import ASRConfig, VoicePaths, rel_to_root
from .utils import ensure_parent, read_jsonl, refuse_overwrite, resolve_artifact_path, write_jsonl

__all__ = ["transcribe_chunks", "export_review", "apply_review"]


def _chunk_paths(paths: VoicePaths) -> list[Path]:
    """Chunk paths from split metadata (accepting legacy absolute paths), or
    a directory glob when metadata is missing."""
    if paths.chunk_metadata.exists():
        metadata = json.loads(paths.chunk_metadata.read_text(encoding="utf-8"))
        return [resolve_artifact_path(row["path"]) for row in metadata.get("clips", [])]
    return sorted(paths.chunks_dir.glob("chunk_*.wav"))


def _select_backend(name: str) -> str:
    order = ["whisperx", "faster-whisper", "whisper"] if name == "auto" else [name]
    for candidate in order:
        try:
            __import__("faster_whisper" if candidate == "faster-whisper" else candidate)
            return candidate
        except Exception:
            continue
    raise SystemExit(
        "No local ASR backend is installed. Install one of:\n"
        "  pip install faster-whisper\n"
        "  pip install whisperx\n"
        "  pip install openai-whisper"
    )


def _build_transcriber(
    backend: str, model_name: str, device: str, compute_type: str, language: str | None
) -> Callable[[Path], str]:
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        def transcribe(path: Path) -> str:
            segments, _info = model.transcribe(str(path), beam_size=5, language=language or None, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()

        return transcribe

    if backend == "whisper":
        import torch
        import whisper

        model = whisper.load_model(model_name, device=device)

        def transcribe(path: Path) -> str:
            result = model.transcribe(
                str(path),
                language=language or None,
                fp16=(device.startswith("cuda") and torch.cuda.is_available()),
            )
            return str(result.get("text", "")).strip()

        return transcribe

    if backend == "whisperx":
        import whisperx

        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language or None)

        def transcribe(path: Path) -> str:
            result = model.transcribe(str(path), batch_size=8)
            return " ".join(segment.get("text", "").strip() for segment in result.get("segments", [])).strip()

        return transcribe

    raise SystemExit(f"Unsupported ASR backend: {backend}")


def transcribe_chunks(paths: VoicePaths, cfg: ASRConfig, force: bool = False) -> list[dict]:
    """Transcribe all chunks to ``paths.transcripts_jsonl`` (step 04).

    Rows are ``{"audio": <project-relative path>, "text": ...}``.
    """
    refuse_overwrite(paths.transcripts_jsonl, force, "transcripts JSONL")
    chunk_paths = _chunk_paths(paths)
    if not chunk_paths:
        raise SystemExit(f"No chunk WAV files found in {paths.chunks_dir}")

    backend = _select_backend(cfg.backend)
    language = cfg.language or None  # empty string means auto-detect
    print(f"Using local ASR backend: {backend} ({cfg.model})")
    transcribe = _build_transcriber(backend, cfg.model, cfg.device, cfg.compute_type, language)

    rows: list[dict] = []
    for path in tqdm(chunk_paths, desc="Transcribing"):
        if not path.exists():
            print(f"WARN missing chunk: {path}")
            continue
        text = transcribe(path)
        rows.append({"audio": rel_to_root(path), "text": text})
        if not text:
            print(f"WARN empty transcript: {path}")

    write_jsonl(paths.transcripts_jsonl, rows, force=force)
    print(f"Wrote transcripts: {paths.transcripts_jsonl}")
    print("Proofread transcripts before training. ASR errors directly degrade fine-tuning quality.")
    return rows


def export_review(paths: VoicePaths, force: bool = False) -> Path:
    """Export transcripts to an editable ``audio\\ttext`` TSV for proofreading."""
    if not paths.transcripts_jsonl.exists():
        raise SystemExit(f"Transcript JSONL does not exist yet: {paths.transcripts_jsonl}")
    refuse_overwrite(paths.review_tsv, force, "review TSV")
    rows = read_jsonl(paths.transcripts_jsonl)
    ensure_parent(paths.review_tsv)
    with paths.review_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio", "text"])
        for row in rows:
            writer.writerow([row.get("audio", ""), row.get("text", "")])
    print(f"Wrote editable review file: {paths.review_tsv}")
    print("Edit the text column, then apply the review to update transcripts.")
    return paths.review_tsv


def apply_review(paths: VoicePaths, force: bool = False) -> int:
    """Rewrite ``paths.transcripts_jsonl`` from the edited review TSV.

    Rows with an empty audio or text cell are skipped with a warning.
    Returns the number of rows applied.
    """
    if not paths.review_tsv.exists():
        raise SystemExit(f"Review TSV does not exist: {paths.review_tsv}")
    refuse_overwrite(paths.transcripts_jsonl, force, "transcripts JSONL")
    rows: list[dict] = []
    with paths.review_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["audio", "text"]:
            raise SystemExit("Review TSV must have exactly these columns: audio, text")
        for line_no, row in enumerate(reader, start=2):
            audio = (row.get("audio") or "").strip()
            text = (row.get("text") or "").strip()
            if not audio or not text:
                empty = "audio" if not audio else "text"
                print(f"WARN skipping review row {line_no}: empty {empty}")
                continue
            rows.append({"audio": audio, "text": text})
    write_jsonl(paths.transcripts_jsonl, rows, force=force)
    print(f"Applied review edits to: {paths.transcripts_jsonl}")
    return len(rows)
