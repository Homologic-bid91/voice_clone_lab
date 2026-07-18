"""Tests for voice_clone_lab.transcribe (ASR backends faked; no GPU/network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import soundfile as sf

import voice_clone_lab.config as config_mod
import voice_clone_lab.transcribe as transcribe_mod
from voice_clone_lab.config import ASRConfig, VoicePaths
from voice_clone_lab.transcribe import apply_review, export_review, transcribe_chunks
from voice_clone_lab.utils import read_jsonl

SR = 24000


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


def _write_transcripts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


SAMPLE_ROWS = [
    {"audio": "data/voices/test/chunks/chunk_0001.wav", "text": "hello world"},
    {"audio": "data/voices/test/chunks/chunk_0002.wav", "text": "second clip"},
]


# ------------------------------------------------------------ export_review


def test_export_review_roundtrip(project_root):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, SAMPLE_ROWS)
    out = export_review(paths)
    assert out == paths.review_tsv
    assert _read_tsv(out) == [["audio", "text"]] + [[r["audio"], r["text"]] for r in SAMPLE_ROWS]


def test_export_review_missing_transcripts(project_root):
    with pytest.raises(SystemExit, match="does not exist"):
        export_review(make_paths(project_root))


def test_export_review_refuses_overwrite(project_root):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, SAMPLE_ROWS)
    export_review(paths)
    with pytest.raises(SystemExit):
        export_review(paths)


# ------------------------------------------------------------ apply_review


def test_apply_review_edits_win(project_root):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, [
        {"audio": "a.wav", "text": "ASR got this wrong"},
        {"audio": "b.wav", "text": "fine"},
    ])
    export_review(paths)
    lines = paths.review_tsv.read_text(encoding="utf-8").splitlines()
    lines[1] = "a.wav\tcorrected by a human"
    paths.review_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    applied = apply_review(paths, force=True)
    assert applied == 2
    assert read_jsonl(paths.transcripts_jsonl) == [
        {"audio": "a.wav", "text": "corrected by a human"},
        {"audio": "b.wav", "text": "fine"},
    ]


def test_apply_review_bad_header(project_root):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, [{"audio": "a.wav", "text": "x"}])
    paths.review_tsv.parent.mkdir(parents=True, exist_ok=True)
    paths.review_tsv.write_text("audio\ttext\textra\na.wav\tx\ty\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="audio, text"):
        apply_review(paths, force=True)


def test_apply_review_warns_and_skips_empty_cells(project_root, capsys):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, [{"audio": "a.wav", "text": "x"}])
    paths.review_tsv.parent.mkdir(parents=True, exist_ok=True)
    paths.review_tsv.write_text("audio\ttext\na.wav\t\n\torphan text\nb.wav\tgood\n", encoding="utf-8")

    applied = apply_review(paths, force=True)
    assert applied == 1
    out = capsys.readouterr().out
    assert out.count("WARN skipping review row") == 2
    assert "empty text" in out
    assert "empty audio" in out
    assert read_jsonl(paths.transcripts_jsonl) == [{"audio": "b.wav", "text": "good"}]


def test_apply_review_refuses_without_force(project_root):
    paths = make_paths(project_root)
    _write_transcripts(paths.transcripts_jsonl, [{"audio": "a.wav", "text": "x"}])
    export_review(paths)
    with pytest.raises(SystemExit):
        apply_review(paths)


# --------------------------------------------------------- transcribe_chunks


def test_transcribe_chunks_writes_relative_paths(speech_like, project_root, monkeypatch, capsys):
    paths = make_paths(project_root)
    paths.chunks_dir.mkdir(parents=True)
    chunk1 = paths.chunks_dir / "chunk_0001.wav"
    chunk2 = paths.chunks_dir / "chunk_0002.wav"
    sf.write(chunk1, speech_like, SR)
    sf.write(chunk2, speech_like, SR)
    # one project-relative (new), one absolute (legacy scripts/), one missing
    metadata = {
        "clips": [
            {"path": "data/voices/test/chunks/chunk_0001.wav"},
            {"path": str(chunk2)},
            {"path": "data/voices/test/chunks/chunk_9999.wav"},
        ]
    }
    paths.chunk_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(transcribe_mod, "_select_backend", lambda name: "fake")
    monkeypatch.setattr(transcribe_mod, "_build_transcriber", lambda *args: lambda path: "fake text")

    rows = transcribe_chunks(paths, ASRConfig())
    assert len(rows) == 2
    for row in rows:
        assert row["text"] == "fake text"
        assert not Path(row["audio"]).is_absolute()
        assert (project_root / row["audio"]).exists()
    assert "WARN missing chunk" in capsys.readouterr().out
    assert read_jsonl(paths.transcripts_jsonl) == rows


def test_transcribe_chunks_without_any_backend(speech_like, project_root):
    paths = make_paths(project_root)
    paths.chunks_dir.mkdir(parents=True)
    sf.write(paths.chunks_dir / "chunk_0001.wav", speech_like, SR)
    with pytest.raises(SystemExit, match="No local ASR backend"):
        transcribe_chunks(paths, ASRConfig(backend="definitely-not-installed"))


def test_transcribe_chunks_no_chunks(project_root):
    paths = make_paths(project_root)
    paths.chunks_dir.mkdir(parents=True)
    with pytest.raises(SystemExit, match="No chunk WAV"):
        transcribe_chunks(paths, ASRConfig())
