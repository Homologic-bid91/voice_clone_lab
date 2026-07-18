"""Shared pytest fixtures: path isolation and synthetic audio (no GPU, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf  # noqa: E402


@pytest.fixture
def tone() -> np.ndarray:
    """One second of a 220 Hz tone at 24 kHz, float32."""
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def speech_like() -> np.ndarray:
    """~1s of speech-like amplitude-modulated noise at 24 kHz (non-silent, unclipped)."""
    rng = np.random.default_rng(0)
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    envelope = 0.2 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    return (envelope * rng.standard_normal(sr)).astype(np.float32)


@pytest.fixture
def write_wav(tmp_path):
    def _write(name: str, samples: np.ndarray, sample_rate: int = 24000) -> Path:
        path = tmp_path / name
        sf.write(path, samples, sample_rate)
        return path

    return _write
