"""Tests for voice_clone_lab.generate — no GPU, no model loads, no network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_clone_lab import generate as gen
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


def make_checkpoint(base: Path, name: str, with_config: bool = True) -> Path:
    ckpt = base / name
    ckpt.mkdir(parents=True, exist_ok=True)
    if with_config:
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
    return ckpt


class FakeTTS:
    """Stands in for Qwen3TTSModel; records generation calls."""

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.custom_voice_calls: list[dict] = []
        self.voice_clone_calls: list[dict] = []

    def generate_custom_voice(self, **kwargs):
        self.custom_voice_calls.append(kwargs)
        return [np.zeros(self.sr // 10, dtype=np.float32)], self.sr

    def generate_voice_clone(self, **kwargs):
        self.voice_clone_calls.append(kwargs)
        return [np.zeros(self.sr // 10, dtype=np.float32)], self.sr


# --- resolve_attn_implementation -------------------------------------------


def test_attn_auto_falls_back_to_sdpa_without_flash(monkeypatch):
    monkeypatch.setattr(gen, "flash_attn_available", lambda: False)
    cfg = Config()
    cfg.qwen.attn_implementation = "auto"
    assert gen.resolve_attn_implementation(cfg) == "sdpa"


def test_attn_auto_picks_flash_when_available(monkeypatch):
    monkeypatch.setattr(gen, "flash_attn_available", lambda: True)
    cfg = Config()
    cfg.qwen.attn_implementation = "auto"
    assert gen.resolve_attn_implementation(cfg) == "flash_attention_2"


def test_attn_explicit_flash_with_no_flash_attn_gives_sdpa():
    cfg = Config()
    cfg.qwen.attn_implementation = "flash_attention_2"
    assert gen.resolve_attn_implementation(cfg, no_flash_attn=True) == "sdpa"


def test_attn_invalid_value_exits():
    cfg = Config()
    cfg.qwen.attn_implementation = "eager"
    with pytest.raises(SystemExit):
        gen.resolve_attn_implementation(cfg)


# --- resolve_checkpoint -----------------------------------------------------


def test_resolve_checkpoint_picks_newest_by_trailing_number(tmp_path):
    paths = make_paths(tmp_path)
    for n in (0, 1, 10):
        make_checkpoint(paths.checkpoints_dir, f"checkpoint-epoch-{n}")
    resolved = gen.resolve_checkpoint(Config(), paths)
    assert resolved.name == "checkpoint-epoch-10"


def test_resolve_checkpoint_explicit_wins_over_config(tmp_path):
    paths = make_paths(tmp_path)
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-1")
    explicit = make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-2")
    cfg = Config()
    cfg.generation.checkpoint = str(paths.checkpoints_dir / "checkpoint-epoch-1")
    assert gen.resolve_checkpoint(cfg, paths, explicit=str(explicit)) == explicit


def test_resolve_checkpoint_empty_dir_exits(tmp_path):
    paths = make_paths(tmp_path)
    paths.checkpoints_dir.mkdir(parents=True)
    with pytest.raises(SystemExit):
        gen.resolve_checkpoint(Config(), paths)


def test_resolve_checkpoint_missing_config_json_exits(tmp_path):
    paths = make_paths(tmp_path)
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-3", with_config=False)
    with pytest.raises(SystemExit):
        gen.resolve_checkpoint(Config(), paths)
    explicit = make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-4", with_config=False)
    with pytest.raises(SystemExit):
        gen.resolve_checkpoint(Config(), paths, explicit=str(explicit))


# --- generate (Content Forge contract) --------------------------------------


def test_generate_writes_wav_with_contract_kwargs(tmp_path, monkeypatch):
    paths = make_paths(tmp_path, speaker="alice")
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-0")
    fake = FakeTTS()
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: fake)

    out = tmp_path / "out" / "line.wav"
    result = gen.generate(Config(), paths, text=" hello world ", out=out)

    assert result == out and out.exists()
    assert len(fake.custom_voice_calls) == 1
    call = fake.custom_voice_calls[0]
    cfg = Config()
    assert call == {
        "text": " hello world ",
        "speaker": "alice",
        "language": cfg.generation.language,
        "do_sample": True,
        "top_k": cfg.generation.top_k,
        "top_p": cfg.generation.top_p,
        "temperature": cfg.generation.temperature,
    }


def test_generate_instruct_and_overrides(tmp_path, monkeypatch):
    paths = make_paths(tmp_path)
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-0")
    fake = FakeTTS()
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: fake)

    gen.generate(
        Config(), paths, text="hi", out=tmp_path / "o.wav",
        instruct="speak slowly", temperature=0.5, top_p=0.8, top_k=10,
    )
    call = fake.custom_voice_calls[0]
    assert call["instruct"] == "speak slowly"
    assert (call["temperature"], call["top_p"], call["top_k"]) == (0.5, 0.8, 10)


def test_generate_refuses_existing_out(tmp_path, monkeypatch):
    paths = make_paths(tmp_path)
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-0")
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: FakeTTS())
    out = tmp_path / "exists.wav"
    out.write_bytes(b"x")
    with pytest.raises(SystemExit):
        gen.generate(Config(), paths, text="hi", out=out)


def test_generate_empty_text_exits(tmp_path):
    with pytest.raises(SystemExit):
        gen.generate(Config(), make_paths(tmp_path), text="  ", out=tmp_path / "o.wav")


# --- generate_batch ---------------------------------------------------------


def write_spec(tmp_path: Path, rows) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(rows), encoding="utf-8")
    return spec


def test_batch_bad_json_exits(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        gen.generate_batch(Config(), make_paths(tmp_path), spec, tmp_path / "out")


def test_batch_missing_keys_exits(tmp_path):
    spec = write_spec(tmp_path, [{"id": "a"}])
    with pytest.raises(SystemExit):
        gen.generate_batch(Config(), make_paths(tmp_path), spec, tmp_path / "out")


def test_batch_overwrite_guard_lists_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: FakeTTS())  # must not be reached
    spec = write_spec(tmp_path, [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}])
    outdir = tmp_path / "out"
    outdir.mkdir()
    existing = outdir / "b.wav"
    existing.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc_info:
        gen.generate_batch(Config(), make_paths(tmp_path), spec, outdir)
    assert "b.wav" in str(exc_info.value)


def test_batch_happy_path_writes_manifest(tmp_path, monkeypatch):
    paths = make_paths(tmp_path, speaker="alice")
    make_checkpoint(paths.checkpoints_dir, "checkpoint-epoch-0")
    fake = FakeTTS()
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: fake)

    spec = write_spec(tmp_path, [{"id": "s1", "text": "first"}, {"id": "s2", "text": "second"}])
    outdir = tmp_path / "out"
    result = gen.generate_batch(Config(), paths, spec, outdir)

    assert result == outdir
    assert (outdir / "s1.wav").exists() and (outdir / "s2.wav").exists()
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in manifest] == ["s1", "s2"]
    assert manifest[0]["src"] == "s1.wav"
    assert manifest[0]["text"] == "first"
    assert manifest[0]["duration"] == pytest.approx(0.1)
    assert len(fake.custom_voice_calls) == 2
    assert fake.custom_voice_calls[0]["speaker"] == "alice"


# --- zeroshot ---------------------------------------------------------------


def test_zeroshot_missing_ref_exits(tmp_path):
    cfg = Config()
    with pytest.raises(SystemExit):
        gen.zeroshot(cfg, make_paths(tmp_path), ref_audio=tmp_path / "nope.wav", text="hi", out=tmp_path / "o.wav")


def test_zeroshot_missing_model_exits(tmp_path, write_wav, tone):
    cfg = Config()
    cfg.qwen.init_model_path = str(tmp_path / "no_model")
    ref = write_wav("ref_src.wav", tone)
    with pytest.raises(SystemExit):
        gen.zeroshot(cfg, make_paths(tmp_path), ref_audio=ref, text="hi", out=tmp_path / "o.wav")


def test_zeroshot_ref_wav_overwrite_guard(tmp_path, write_wav, tone):
    ref = write_wav("ref_src.wav", tone)
    out = tmp_path / "gen" / "clip.wav"
    out.parent.mkdir(parents=True)
    derived = out.with_name("clip_ref.wav")
    derived.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc_info:
        gen.zeroshot(Config(), make_paths(tmp_path), ref_audio=ref, text="hi", out=out)
    assert "clip_ref.wav" in str(exc_info.value)


def test_zeroshot_happy_path(tmp_path, write_wav, tone, monkeypatch):
    ref = write_wav("ref_src.wav", tone)
    fake = FakeTTS()
    monkeypatch.setattr(gen, "load_model", lambda *a, **k: fake)
    monkeypatch.setattr(gen, "_convert_ref_to_wav", lambda src, dst, sr: dst)

    out = tmp_path / "gen" / "clip.wav"
    result = gen.zeroshot(Config(), make_paths(tmp_path), ref_audio=ref, text="hello", out=out)

    assert result == out and out.exists()
    call = fake.voice_clone_calls[0]
    assert call["x_vector_only_mode"] is True
    assert call["ref_text"] is None
    assert call["ref_audio"].endswith("clip_ref.wav")
    assert call["do_sample"] is True
