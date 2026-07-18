"""Tests for voice_clone_lab.dataset (synthetic wavs on disk, no ASR)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import voice_clone_lab.config as config_mod
from voice_clone_lab.config import AudioConfig, VoicePaths
from voice_clone_lab.dataset import build_dataset
from voice_clone_lab.utils import read_jsonl

SR = 24000

EXPECTED_STAT_KEYS = {
    "clips",
    "total_minutes",
    "min_duration_seconds",
    "max_duration_seconds",
    "mean_duration_seconds",
    "rejected_clips",
    "rejected",
    "reference_audio",
    "reference_stats",
    "path_mode",
}


def make_paths(root: Path) -> VoicePaths:
    voice = root / "data" / "voices" / "test"
    dataset = voice / "dataset"
    return VoicePaths(
        speaker="test",
        voice_dir=voice,
        raw_dir=voice / "raw",
        extracted_audio=voice / "extracted" / "test.wav",
        cleaned_audio=voice / "cleaned" / "test_clean.wav",
        chunks_dir=voice / "chunks",
        chunk_metadata=voice / "chunks" / "metadata.json",
        transcripts_jsonl=voice / "transcripts" / "transcripts.jsonl",
        review_tsv=voice / "transcripts" / "transcripts_review.tsv",
        reference_audio=voice / "reference" / "ref.wav",
        dataset_dir=dataset,
        train_raw_jsonl=dataset / "train_raw.jsonl",
        train_with_codes_jsonl=dataset / "train_with_codes.jsonl",
        dataset_stats=dataset / "dataset_stats.json",
        checkpoints_dir=root / "outputs" / "checkpoints" / "test",
        generated_dir=root / "outputs" / "generated" / "test",
    )


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "ROOT", tmp_path)
    return tmp_path


def _seconds(speech_like: np.ndarray, n: float) -> np.ndarray:
    return np.tile(speech_like, int(np.ceil(n)))[: int(n * SR)]


def _write_chunk(paths: VoicePaths, name: str, samples: np.ndarray) -> None:
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    sf.write(paths.chunks_dir / name, samples, SR)


@pytest.fixture
def voice(project_root, speech_like):
    """Chunks of 8s/6s/4s plus transcripts with one missing file and one empty text."""
    paths = make_paths(project_root)
    rows = []
    for name, seconds in [("chunk_a.wav", 8), ("chunk_b.wav", 6), ("chunk_c.wav", 4)]:
        _write_chunk(paths, name, _seconds(speech_like, seconds))
        rows.append({"audio": f"data/voices/test/chunks/{name}", "text": f"transcript for {name}"})
    rows.append({"audio": "data/voices/test/chunks/missing.wav", "text": "gone"})
    rows.append({"audio": "data/voices/test/chunks/chunk_a.wav", "text": ""})
    paths.transcripts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with paths.transcripts_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return paths


def test_build_dataset_rows_stats_and_reference(voice, project_root, capsys):
    ft_dir = project_root / "third_party" / "Qwen3-TTS" / "finetuning"
    stats = build_dataset(voice, AudioConfig(), SR, qwen_finetuning_dir=ft_dir)

    assert EXPECTED_STAT_KEYS <= set(stats)
    assert stats["clips"] == 3
    assert stats["rejected_clips"] == 2
    assert sorted(r["reason"] for r in stats["rejected"]) == ["audio file missing", "empty transcript"]
    assert stats["total_minutes"] == pytest.approx(18.0 / 60.0, abs=0.01)
    assert stats["min_duration_seconds"] == pytest.approx(4.0, abs=0.01)
    assert stats["max_duration_seconds"] == pytest.approx(8.0, abs=0.01)
    assert stats["path_mode"] == "qwen-relative"

    # auto-selection picks the 8s chunk: closest to target_chunk_seconds (8.0)
    assert voice.reference_audio.exists()
    assert stats["reference_stats"]["duration_seconds"] == pytest.approx(8.0, abs=0.01)

    rows = read_jsonl(voice.train_raw_jsonl)
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {"audio", "text", "ref_audio"}
        resolved_audio = (ft_dir / row["audio"]).resolve()
        assert resolved_audio.parent == voice.chunks_dir.resolve()
        assert resolved_audio.exists()
        assert (ft_dir / row["ref_audio"]).resolve() == voice.reference_audio.resolve()
    expected = os.path.relpath((voice.chunks_dir / "chunk_a.wav").resolve(), ft_dir)
    assert rows[0]["audio"] == expected

    # 18s total is far below the 5-minute default -> soft warning
    assert "WARN: only" in capsys.readouterr().out


def test_build_dataset_project_relative_and_absolute_modes(voice):
    stats = build_dataset(voice, AudioConfig(), SR, path_mode="project-relative")
    rows = read_jsonl(voice.train_raw_jsonl)
    assert rows[0]["audio"] == "data/voices/test/chunks/chunk_a.wav"
    assert stats["path_mode"] == "project-relative"

    stats = build_dataset(voice, AudioConfig(), SR, path_mode="absolute", force=True)
    rows = read_jsonl(voice.train_raw_jsonl)
    assert Path(rows[0]["audio"]).is_absolute()
    assert Path(rows[0]["ref_audio"]).is_absolute()


def test_build_dataset_reuses_existing_reference(voice, speech_like, capsys):
    voice.reference_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(voice.reference_audio, _seconds(speech_like, 7), SR)
    stats = build_dataset(voice, AudioConfig(), SR)
    assert stats["reference_stats"]["duration_seconds"] == pytest.approx(7.0, abs=0.01)
    assert "Using existing reference audio" in capsys.readouterr().out


def test_build_dataset_copies_source_ref(voice, speech_like, tmp_path, capsys):
    src = tmp_path / "my_ref.wav"
    sf.write(src, _seconds(speech_like, 6), SR)
    stats = build_dataset(voice, AudioConfig(), SR, source_ref=src)
    assert "Copied reference audio" in capsys.readouterr().out
    assert voice.reference_audio.exists()
    assert stats["reference_stats"]["duration_seconds"] == pytest.approx(6.0, abs=0.01)


def test_build_dataset_min_minutes_warning_toggle(voice, capsys):
    build_dataset(voice, AudioConfig(), SR, min_minutes=0.0)
    assert "WARN: only" not in capsys.readouterr().out
    build_dataset(voice, AudioConfig(), SR, min_minutes=10.0, force=True)
    assert "WARN: only" in capsys.readouterr().out


def test_build_dataset_refuses_overwrite(voice):
    build_dataset(voice, AudioConfig(), SR)
    with pytest.raises(SystemExit):
        build_dataset(voice, AudioConfig(), SR)


def test_build_dataset_no_valid_rows(voice):
    voice.transcripts_jsonl.write_text(
        json.dumps({"audio": "data/voices/test/chunks/missing.wav", "text": "x"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="No valid transcript rows"):
        build_dataset(voice, AudioConfig(), SR)


def test_build_dataset_missing_transcripts(project_root):
    with pytest.raises(SystemExit, match="Transcripts file does not exist"):
        build_dataset(make_paths(project_root), AudioConfig(), SR)
