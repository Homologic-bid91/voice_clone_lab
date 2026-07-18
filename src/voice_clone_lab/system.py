"""System dependency checks for the fine-tuning pipeline (port of 00_check_system.py).

``check_system`` returns structured ``CheckResult`` rows; ``format_report``
renders them as aligned `[OK]/[WARN]/[FAIL]` lines with a summary. Optional
dependencies are probed with importlib.util.find_spec so the check stays cheap;
torch is imported for real because the CUDA build matters.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, Config
from .utils import command_exists, run

__all__ = ["CheckResult", "check_system", "format_report"]


@dataclass
class CheckResult:
    name: str
    status: str  # "OK" | "WARN" | "FAIL"
    detail: str


def _spec_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _probe(module: str, dist: str | None = None) -> str:
    """Version string for an installed module, else 'installed'."""
    try:
        return importlib.metadata.version(dist or module)
    except importlib.metadata.PackageNotFoundError:
        return "installed"


def _vram_guidance(vram_gb: float) -> str:
    if vram_gb >= 32:
        return "batch_size 4-8 OK"
    if vram_gb >= 24:
        return "use --batch_size 2"
    if vram_gb >= 16:
        return "training tight, try --batch_size 2; generation fine"
    return "generation only, training unlikely without code changes"


def check_system(cfg: Config, min_free_gb: int | None = None) -> list[CheckResult]:
    """Run all local dependency checks and return structured results."""
    results: list[CheckResult] = []
    add = results.append

    python = platform.python_version()
    if sys.version_info < (3, 12):
        add(CheckResult(
            "python", "WARN",
            f"{python} — Qwen3-TTS docs recommend 3.12: conda create -n qwen3-tts python=3.12 -y",
        ))
    else:
        add(CheckResult("python", "OK", python))

    if command_exists("ffmpeg"):
        add(CheckResult("ffmpeg", "OK", str(shutil.which("ffmpeg"))))
    else:
        add(CheckResult("ffmpeg", "FAIL", "missing. Install with: sudo apt-get install ffmpeg"))
    for binary in ("ffprobe", "git"):
        if command_exists(binary):
            add(CheckResult(binary, "OK", str(shutil.which(binary))))
        else:
            add(CheckResult(binary, "WARN", f"missing. Install with: sudo apt-get install {'ffmpeg' if binary == 'ffprobe' else 'git'}"))

    try:
        import torch
    except Exception as exc:
        add(CheckResult("torch", "FAIL", f"missing ({exc.__class__.__name__}). Install with: pip install torch torchaudio"))
    else:
        cuda_build = torch.version.cuda
        if cuda_build is None:
            add(CheckResult("torch", "FAIL", f"{torch.__version__} built without CUDA. Install a CUDA-enabled PyTorch build."))
        elif not torch.cuda.is_available():
            add(CheckResult(
                "torch", "FAIL",
                f"{torch.__version__} (CUDA {cuda_build}) but CUDA is unavailable. Check NVIDIA drivers with nvidia-smi.",
            ))
        else:
            add(CheckResult("torch", "OK", f"{torch.__version__} (CUDA build {cuda_build})"))
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                vram_gb = props.total_memory / 1024**3
                add(CheckResult(f"GPU {idx}", "OK", f"{props.name}, {vram_gb:.1f} GB VRAM — {_vram_guidance(vram_gb)}"))

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    required = min_free_gb if min_free_gb is not None else cfg.system.min_free_gb
    if free_gb < required:
        add(CheckResult("disk free", "WARN", f"{free_gb:.1f} GB free at project root, below the {required} GB recommended for models/checkpoints"))
    else:
        add(CheckResult("disk free", "OK", f"{free_gb:.1f} GB free at project root"))

    if _spec_present("flash_attn"):
        add(CheckResult("flash_attn", "OK", _probe("flash_attn", "flash-attn")))
    else:
        add(CheckResult("flash_attn", "WARN", "not installed — falls back to sdpa. Install with: pip install flash-attn --no-build-isolation"))

    repo = cfg.qwen_repo_path()
    sft = repo / "finetuning" / "sft_12hz.py"
    if not sft.exists():
        add(CheckResult("Qwen3-TTS repo", "WARN", f"not vendored at {repo} — run vcl setup"))
    else:
        head = run(["git", "rev-parse", "HEAD"], cwd=repo)
        if head == cfg.qwen.repo_commit:
            add(CheckResult("Qwen3-TTS repo", "OK", f"{repo} @ {head[:12]}"))
        else:
            add(CheckResult(
                "Qwen3-TTS repo", "WARN",
                f"commit {head or 'unknown'} != pinned {cfg.qwen.repo_commit} — run vcl setup",
            ))
        if "--attn_implementation" in sft.read_text(encoding="utf-8", errors="replace"):
            add(CheckResult("sft_12hz patch", "OK", "patches/sft_12hz.patch applied"))
        else:
            add(CheckResult("sft_12hz patch", "WARN", "local patch not applied — run vcl setup"))

    if _spec_present("faster_whisper"):
        add(CheckResult("faster_whisper", "OK", _probe("faster_whisper", "faster-whisper")))
    else:
        add(CheckResult("faster_whisper", "WARN", "not installed — transcription falls back to whisperx/openai-whisper. Install with: pip install faster-whisper"))
    for module, dist, note in [
        ("whisperx", "whisperx", "optional ASR backend (word timestamps)"),
        ("webrtcvad", "webrtcvad", "optional; improves silence-based splitting"),
        ("noisereduce", "noisereduce", "optional; enables denoising"),
    ]:
        if _spec_present(module):
            add(CheckResult(module, "OK", f"{_probe(module, dist)} ({note})"))
        else:
            add(CheckResult(module, "WARN", f"not installed — {note}"))

    model_dir = cfg.init_model_path()
    if model_dir.is_dir():
        add(CheckResult("init model", "OK", str(model_dir)))
    else:
        add(CheckResult(
            "init model", "WARN",
            f"not downloaded: {model_dir}. Get it with: huggingface-cli download {cfg.qwen.init_model_id} --local-dir {model_dir}",
        ))

    return results


def format_report(results: list[CheckResult]) -> str:
    """Render results as aligned [OK]/[WARN]/[FAIL] lines plus a summary."""
    width = max((len(r.name) for r in results), default=0)
    lines = [f"{f'[{r.status}]'.ljust(7)} {r.name.ljust(width)}  {r.detail}" for r in results]
    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines.append("")
    lines.append(f"Summary: {counts['OK']} OK, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    if counts["FAIL"]:
        lines.append("System check found missing required pieces. Fix the [FAIL] items and re-run.")
    else:
        lines.append("System check passed for required local pieces.")
    return "\n".join(lines)
