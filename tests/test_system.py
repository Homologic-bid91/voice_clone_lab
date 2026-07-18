"""Tests for voice_clone_lab.system — no GPU assumptions, no network."""

from __future__ import annotations

import shutil

from voice_clone_lab.config import Config
from voice_clone_lab.system import CheckResult, check_system, format_report


def test_check_system_runs_to_completion():
    results = check_system(Config())
    assert isinstance(results, list) and len(results) > 0
    assert all(isinstance(r, CheckResult) for r in results)
    assert all(r.status in {"OK", "WARN", "FAIL"} for r in results)
    assert all(r.name and r.detail for r in results)


def test_check_system_missing_ffmpeg_is_fail(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    results = check_system(Config())
    ffmpeg = [r for r in results if r.name == "ffmpeg"]
    assert len(ffmpeg) == 1
    assert ffmpeg[0].status == "FAIL"


def test_vram_guidance_boundaries():
    from voice_clone_lab.system import _vram_guidance

    assert _vram_guidance(48) == "batch_size 4-8 OK"
    assert _vram_guidance(32) == "batch_size 4-8 OK"
    assert _vram_guidance(24) == "use --batch_size 2"
    assert _vram_guidance(16) == "training tight, try --batch_size 2; generation fine"
    assert _vram_guidance(8) == "generation only, training unlikely without code changes"


def test_format_report_alignment_and_summary():
    results = [
        CheckResult("python", "OK", "3.12.0"),
        CheckResult("flash_attn", "WARN", "not installed"),
        CheckResult("ffmpeg", "FAIL", "missing"),
    ]
    report = format_report(results)
    assert "[OK]" in report and "[WARN]" in report and "[FAIL]" in report
    assert "python" in report and "3.12.0" in report
    assert "Summary: 1 OK, 1 WARN, 1 FAIL" in report
    # Name column is aligned: tag field is 8 chars, name field pads to the longest name.
    width = len("flash_attn")
    for line, name in zip(report.splitlines()[:3], ["python", "flash_attn", "ffmpeg"]):
        assert line[8 : 8 + width] == name.ljust(width)


def test_format_report_fail_footer():
    report = format_report([CheckResult("x", "FAIL", "bad")])
    assert "missing required pieces" in report
    report_ok = format_report([CheckResult("x", "OK", "fine")])
    assert "passed" in report_ok
