"""Tests for voice_clone_lab.train — no training, no downloads, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_clone_lab import train as train_mod
from voice_clone_lab.config import Config, VoicePaths


def make_paths(tmp_path: Path, speaker: str = "test") -> VoicePaths:
    return VoicePaths(
        speaker=speaker,
        voice_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        extracted_audio=tmp_path / "extracted.wav",
        cleaned_audio=tmp_path / "clean.wav",
        chunks_dir=tmp_path / "chunks",
        chunk_metadata=tmp_path / "chunks" / "metadata.json",
        transcripts_jsonl=tmp_path / "transcripts.jsonl",
        review_tsv=tmp_path / "review.tsv",
        reference_audio=tmp_path / "ref.wav",
        dataset_dir=tmp_path / "dataset",
        train_raw_jsonl=tmp_path / "dataset" / "train_raw.jsonl",
        train_with_codes_jsonl=tmp_path / "dataset" / "train_with_codes.jsonl",
        dataset_stats=tmp_path / "dataset" / "dataset_stats.json",
        checkpoints_dir=tmp_path / "checkpoints",
        generated_dir=tmp_path / "generated",
    )


def prepare_train_preconditions(tmp_path: Path, monkeypatch) -> VoicePaths:
    """Make train() reach its force guard without touching any subprocess."""
    paths = make_paths(tmp_path)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.train_with_codes_jsonl.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_mod, "resolve_init_model", lambda cfg, allow_download=False: tmp_path)
    monkeypatch.setattr(train_mod, "resolve_attn_implementation", lambda cfg: "sdpa")
    return paths


# --- train force guard ------------------------------------------------------


def test_train_force_outside_checkpoints_refused(tmp_path, monkeypatch):
    paths = prepare_train_preconditions(tmp_path, monkeypatch)
    output_dir = tmp_path / "somewhere_else"
    output_dir.mkdir()
    (output_dir / "checkpoint-epoch-0").mkdir()
    with pytest.raises(SystemExit) as exc_info:
        train_mod.train(Config(), paths, output_dir=output_dir, force=True)
    assert "outside outputs/checkpoints" in str(exc_info.value)
    assert output_dir.exists()  # untouched


def test_train_nonempty_output_dir_without_force_refused(tmp_path, monkeypatch):
    paths = prepare_train_preconditions(tmp_path, monkeypatch)
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    (output_dir / "old.bin").write_bytes(b"x")
    with pytest.raises(SystemExit) as exc_info:
        train_mod.train(Config(), paths, output_dir=output_dir)
    assert "Refusing to overwrite" in str(exc_info.value)


def test_train_missing_train_jsonl_exits(tmp_path):
    paths = make_paths(tmp_path)
    with pytest.raises(SystemExit):
        train_mod.train(Config(), paths, output_dir=tmp_path / "out")


# --- resolve_init_model -----------------------------------------------------


def test_resolve_init_model_existing_dir(tmp_path):
    cfg = Config()
    cfg.qwen.init_model_path = str(tmp_path)
    assert train_mod.resolve_init_model(cfg) == tmp_path.resolve()


def test_resolve_init_model_missing_dir_exits_with_download_command(tmp_path):
    cfg = Config()
    target = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
    cfg.qwen.init_model_path = str(target)
    with pytest.raises(SystemExit) as exc_info:
        train_mod.resolve_init_model(cfg)
    message = str(exc_info.value)
    assert f"huggingface-cli download {cfg.qwen.init_model_id} --local-dir {target}" in message


# --- prepare_codes ----------------------------------------------------------


def test_prepare_codes_missing_input_exits(tmp_path):
    paths = make_paths(tmp_path)
    with pytest.raises(SystemExit):
        train_mod.prepare_codes(Config(), paths)


def test_prepare_codes_refuses_existing_output(tmp_path):
    paths = make_paths(tmp_path)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.train_raw_jsonl.write_text("{}\n", encoding="utf-8")
    paths.train_with_codes_jsonl.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_mod.prepare_codes(Config(), paths)


def test_default_tokenizer_prefers_bundled_speech_tokenizer():
    # The real models/Qwen3-TTS-12Hz-1.7B-Base ships speech_tokenizer/.
    tokenizer = train_mod._default_tokenizer_model_path(Config())
    assert tokenizer.endswith("speech_tokenizer")


def test_default_tokenizer_falls_back_to_hf_id(tmp_path):
    cfg = Config()
    cfg.qwen.init_model_path = str(tmp_path)  # exists but has no speech_tokenizer/
    assert train_mod._default_tokenizer_model_path(cfg) == train_mod.DEFAULT_TOKENIZER_ID


# --- write_train_metadata ---------------------------------------------------


def test_write_train_metadata_keys(tmp_path):
    paths = make_paths(tmp_path)
    out = tmp_path / "ckpt"
    out.mkdir()
    path = train_mod.write_train_metadata(
        out,
        cfg=Config(),
        paths=paths,
        init_model_path=tmp_path,
        train_jsonl=paths.train_with_codes_jsonl,
        batch_size=2,
        lr=2e-6,
        epochs=1,
        speaker="test",
        attn_implementation="sdpa",
    )
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "timestamp_utc", "python", "platform", "qwen_repo_commit", "qwen_repo_path",
        "init_model_path", "init_model_id", "train_jsonl", "output_dir",
        "batch_size", "lr", "epochs", "speaker_name", "attn_implementation",
    ):
        assert key in meta, key
    assert meta["batch_size"] == 2
    assert meta["epochs"] == 1
    assert meta["speaker_name"] == "test"
    assert meta["attn_implementation"] == "sdpa"
    # torch is installed in this environment; qwen_tts may or may not import.
    assert "torch_version" in meta or "torch_error" in meta
    assert "qwen_tts_version" in meta or "qwen_tts_error" in meta
