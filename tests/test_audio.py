"""Tests for voice_clone_lab.audio (no GPU, no network, ffmpeg mocked out)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import voice_clone_lab.audio as audio_mod
import voice_clone_lab.config as config_mod
from voice_clone_lab.audio import (
    audio_stats,
    clean_audio,
    extract_audio,
    load_audio,
    split_audio,
    warning_lines,
)
from voice_clone_lab.config import AudioConfig, VoicePaths

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
    """Point the project ROOT at tmp_path so artifact paths come out relative."""
    monkeypatch.setattr(config_mod, "ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------- audio_stats


def test_audio_stats_tone(tone):
    stats = audio_stats(tone, SR)
    assert stats["duration_seconds"] == pytest.approx(1.0)
    assert stats["peak"] == pytest.approx(0.3, abs=1e-3)
    assert stats["clipped_samples"] == 0
    assert -20.0 < stats["rms_dbfs"] < -10.0
    assert stats["sample_rate"] == SR


def test_audio_stats_silence():
    stats = audio_stats(np.zeros(SR, dtype=np.float32), SR)
    assert stats["rms_dbfs"] < -100.0
    assert stats["peak"] == 0.0
    assert stats["clipped_samples"] == 0
    assert any("very quiet" in w for w in warning_lines(stats))


def _legacy_frame_db(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """The pure-Python frame-RMS loop from scripts/common.py, for comparison."""
    frame = max(1, int(sample_rate * 0.03))
    hop = max(1, int(sample_rate * 0.01))
    frame_rms: list[float] = []
    for start in range(0, max(1, len(samples) - frame + 1), hop):
        segment = samples[start : start + frame]
        if len(segment):
            frame_rms.append(float(np.sqrt(np.mean(np.square(segment)))))
    return 20.0 * np.log10(np.maximum(np.array(frame_rms), 1e-12))


@pytest.mark.parametrize("n", [SR, 17000, 500])  # exact second, odd tail, shorter than one frame
def test_audio_stats_frame_math_matches_legacy(speech_like, n):
    samples = np.tile(speech_like, 2)[:n]
    stats = audio_stats(samples, SR)
    frame_db = _legacy_frame_db(samples, SR)
    noise = float(np.percentile(frame_db, 10))
    speech = float(np.percentile(frame_db, 90))
    assert stats["noise_floor_dbfs"] == pytest.approx(noise, abs=1e-4)
    assert stats["speech_floor_dbfs"] == pytest.approx(speech, abs=1e-4)
    assert stats["estimated_snr_db"] == pytest.approx(speech - noise, abs=1e-4)


def test_warning_lines_clipping():
    samples = np.ones(SR, dtype=np.float32)
    stats = audio_stats(samples, SR)
    assert stats["clipped_samples"] == SR
    assert any("clipping" in w for w in warning_lines(stats))


# ---------------------------------------------------------------- load_audio


def test_load_audio_downmixes_and_resamples(tmp_path, tone):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([tone, tone]), SR)
    audio = load_audio(path, SR)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == SR
    assert len(load_audio(path, 16000)) == 16000


# ------------------------------------------------------------ extract_audio


class _FakeFFmpeg:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    monkeypatch.setattr(audio_mod, "command_exists", lambda name: True)
    fake = _FakeFFmpeg()
    monkeypatch.setattr(audio_mod.subprocess, "run", fake)
    return fake


def test_extract_audio_builds_expected_command(write_wav, tone, tmp_path, fake_ffmpeg):
    src = write_wav("input.wav", tone)
    out = tmp_path / "nested" / "extracted.wav"
    extract_audio(src, out, SR)
    (cmd,) = fake_ffmpeg.calls
    assert audio_mod.FFMPEG_FILTER_CHAIN in cmd
    assert cmd[cmd.index("-ar") + 1] == str(SR)
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-map") + 1] == "0:a:0"
    assert "-n" in cmd  # no force -> ffmpeg must not overwrite either


def test_extract_audio_force_passes_y(write_wav, tone, tmp_path, fake_ffmpeg):
    src = write_wav("input.wav", tone)
    extract_audio(src, tmp_path / "out.wav", SR, force=True)
    (cmd,) = fake_ffmpeg.calls
    assert "-y" in cmd


def test_extract_audio_refuses_existing_output(write_wav, tone, tmp_path, fake_ffmpeg):
    src = write_wav("input.wav", tone)
    out = write_wav("existing.wav", tone)
    with pytest.raises(SystemExit):
        extract_audio(src, out, SR)
    assert fake_ffmpeg.calls == []


def test_extract_audio_missing_input(tmp_path, fake_ffmpeg):
    with pytest.raises(SystemExit, match="does not exist"):
        extract_audio(tmp_path / "nope.wav", tmp_path / "out.wav", SR)


def test_extract_audio_ffmpeg_failure(write_wav, tone, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(audio_mod.subprocess, "run", _FakeFFmpeg(returncode=1, stderr="boom"))
    src = write_wav("input.wav", tone)
    with pytest.raises(SystemExit, match="ffmpeg failed"):
        extract_audio(src, tmp_path / "out.wav", SR)


# -------------------------------------------------------------- clean_audio


def test_clean_audio_trims_long_silence(speech_like, write_wav, tmp_path):
    silence = np.zeros(3 * SR, dtype=np.float32)
    src = write_wav("raw.wav", np.concatenate([speech_like, silence, speech_like]))
    out = tmp_path / "clean.wav"
    result = clean_audio(src, out, AudioConfig(), SR)
    assert out.exists()
    assert set(result) == {"before", "after", "warnings"}
    assert result["before"]["duration_seconds"] == pytest.approx(5.0)
    # ~3 s of silence removed, keeping padding + one joined gap
    assert 2.0 < result["after"]["duration_seconds"] < 3.2
    assert len(load_audio(out, SR)) == pytest.approx(result["after"]["duration_seconds"] * SR, abs=2)


def test_clean_audio_refuses_overwrite(speech_like, write_wav, tmp_path):
    src = write_wav("raw.wav", speech_like)
    out = write_wav("clean.wav", speech_like)
    with pytest.raises(SystemExit):
        clean_audio(src, out, AudioConfig(), SR)


def test_clean_audio_denoise_guard_without_noisereduce(speech_like, write_wav, tmp_path, capsys):
    try:
        import noisereduce  # noqa: F401

        pytest.skip("noisereduce is installed; the missing-package guard is not exercised")
    except ImportError:
        pass
    src = write_wav("raw.wav", speech_like)
    out = tmp_path / "clean.wav"
    clean_audio(src, out, AudioConfig(), SR, denoise=True)
    assert out.exists()
    assert "noisereduce" in capsys.readouterr().out


# -------------------------------------------------------------- split_audio


def _split_cfg(**overrides) -> AudioConfig:
    values = {"min_chunk_seconds": 0.5}
    values.update(overrides)
    return AudioConfig(**values)


def test_split_audio_writes_relative_clip_paths(speech_like, write_wav, project_root):
    paths = make_paths(project_root)
    silence = np.zeros(2 * SR, dtype=np.float32)
    src = write_wav("clean.wav", np.concatenate([speech_like, silence, speech_like]))
    metadata = split_audio(src, paths, _split_cfg(), SR)

    assert metadata["clips"], "expected at least one accepted chunk"
    for clip in metadata["clips"]:
        assert not Path(clip["path"]).is_absolute()
        assert (project_root / clip["path"]).exists()
    assert (paths.chunks_dir / "chunk_0001.wav").exists()
    assert paths.chunk_metadata.exists()
    on_disk = json.loads(paths.chunk_metadata.read_text(encoding="utf-8"))
    assert on_disk["clips"] == metadata["clips"]
    assert on_disk["sample_rate"] == SR
    assert "rejected" in metadata


def test_split_audio_rejects_silent_file(write_wav, project_root):
    paths = make_paths(project_root)
    src = write_wav("silence.wav", np.zeros(4 * SR, dtype=np.float32))
    metadata = split_audio(src, paths, _split_cfg(), SR)
    assert metadata["clips"] == []
    assert len(metadata["rejected"]) >= 1
    assert paths.chunk_metadata.exists()


def test_split_audio_refuses_rerun_without_force(speech_like, write_wav, project_root):
    paths = make_paths(project_root)
    src = write_wav("clean.wav", np.concatenate([speech_like, speech_like]))
    split_audio(src, paths, _split_cfg(), SR)
    with pytest.raises(SystemExit):
        split_audio(src, paths, _split_cfg(), SR)
    metadata = split_audio(src, paths, _split_cfg(), SR, force=True)
    assert metadata["clips"]
