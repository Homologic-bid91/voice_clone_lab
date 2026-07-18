"""Vendor setup, tokenizer code preparation, and fine-tuning wrappers.

Replaces the bash pipeline (06_prepare_qwen_data.sh, 07_train_qwen.sh):
``setup_vendor`` clones/pins/patches the Qwen3-TTS repo, ``prepare_codes``
runs the upstream prepare_data.py, and ``train`` runs sft_12hz.py with
overwrite guards, provenance files, and a tee'd training log.
"""

from __future__ import annotations

import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, ROOT, Config, VoicePaths, project_path
from .generate import resolve_attn_implementation
from .utils import ensure_parent, refuse_overwrite, run, write_json

__all__ = [
    "setup_vendor",
    "resolve_init_model",
    "prepare_codes",
    "write_train_metadata",
    "train",
]

PATCH_PATH = ROOT / "patches" / "sft_12hz.patch"
DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def _run_or_exit(cmd: list, cwd: Path | None = None, what: str | None = None) -> None:
    """Run a command with live (inherited) stdio; SystemExit on non-zero."""
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        label = what or " ".join(cmd)
        raise SystemExit(f"Command failed (exit {result.returncode}): {label}")


def _git_apply_check(repo: Path, patch: Path, reverse: bool = False) -> bool:
    cmd = ["git", "apply", "--check"]
    if reverse:
        cmd.append("-R")
    cmd.append(str(patch))
    return subprocess.run(
        cmd, cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def setup_vendor(cfg: Config) -> None:
    """Clone, pin, patch, and pip-install the Qwen3-TTS repo (idempotent)."""
    repo = cfg.qwen_repo_path()
    sft = repo / "finetuning" / "sft_12hz.py"
    if sft.exists():
        print(f"Qwen3-TTS repo present: {repo}")
        head = run(["git", "rev-parse", "HEAD"], cwd=repo)
        if head != cfg.qwen.repo_commit:
            print(f"Repo commit {head or 'unknown'} != pinned {cfg.qwen.repo_commit}; checking out the pinned commit.")
            result = subprocess.run(["git", "checkout", cfg.qwen.repo_commit], cwd=repo)
            if result.returncode != 0:
                _run_or_exit(["git", "fetch", "origin"], cwd=repo, what="git fetch origin")
                _run_or_exit(["git", "checkout", cfg.qwen.repo_commit], cwd=repo, what="git checkout pinned commit")
    else:
        if repo.exists():
            raise SystemExit(
                f"{repo} exists but finetuning/sft_12hz.py is missing; fix or remove the checkout and re-run."
            )
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {cfg.qwen.repo_url} -> {repo}")
        _run_or_exit(["git", "clone", cfg.qwen.repo_url, str(repo)], what="git clone Qwen3-TTS")
        print(f"Checking out pinned commit {cfg.qwen.repo_commit}")
        _run_or_exit(["git", "checkout", cfg.qwen.repo_commit], cwd=repo, what="git checkout pinned commit")

    if _git_apply_check(repo, PATCH_PATH):
        print(f"Applying patch: {PATCH_PATH.name}")
        _run_or_exit(["git", "apply", str(PATCH_PATH)], cwd=repo, what=f"git apply {PATCH_PATH.name}")
    elif _git_apply_check(repo, PATCH_PATH, reverse=True):
        print(f"Patch already applied: {PATCH_PATH.name}")
    else:
        raise SystemExit(
            f"{PATCH_PATH} neither applies cleanly nor is already applied to {repo}. Fix the checkout manually."
        )

    print(f"Installing qwen-tts (editable) from {repo}")
    _run_or_exit([sys.executable, "-m", "pip", "install", "-e", str(repo)], what="pip install -e Qwen3-TTS")
    print("Vendor setup complete.")


def resolve_init_model(cfg: Config, allow_download: bool = False) -> Path:
    """Resolve the init model to a local directory, optionally downloading it.

    The upstream training script copies the init model dir into each
    checkpoint, so an HF model id alone is not enough.
    """
    value = cfg.qwen.init_model_path
    model_id = cfg.qwen.init_model_id
    if value.startswith("Qwen/"):  # HF-id-shaped value instead of a local path
        model_id = value
        target = ROOT / "models" / value.rsplit("/", 1)[-1]
    else:
        target = project_path(value)
    if target.is_dir():
        return target.resolve()

    download_cmd = f"huggingface-cli download {model_id} --local-dir {target}"
    if not allow_download:
        raise SystemExit(
            f"Model directory does not exist: {target}\n"
            "The official training script needs a local directory for checkpoint copying.\n"
            "Download once with:\n"
            f"  {download_cmd}\n"
            "Then re-run, or pass --allow-download."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {target}")
    cli = shutil.which("huggingface-cli")
    cmd = [cli] if cli else [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli"]
    _run_or_exit(cmd + ["download", model_id, "--local-dir", str(target)], what=f"download {model_id}")
    if not target.is_dir():
        raise SystemExit(f"Download finished but model directory is still missing: {target}")
    return target.resolve()


def _default_tokenizer_model_path(cfg: Config) -> str:
    """Bundled speech_tokenizer/ inside the init model dir, else the HF id."""
    bundled = cfg.init_model_path() / "speech_tokenizer"
    if bundled.is_dir():
        return str(bundled)
    print(f"WARN: {bundled} not found; prepare_data.py will use {DEFAULT_TOKENIZER_ID} from HF.")
    return DEFAULT_TOKENIZER_ID


def prepare_codes(
    cfg: Config,
    paths: VoicePaths,
    device: str | None = None,
    tokenizer_model_path: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Run upstream prepare_data.py: train_raw.jsonl -> train_with_codes.jsonl."""
    input_jsonl = paths.train_raw_jsonl
    output_jsonl = paths.train_with_codes_jsonl
    if not input_jsonl.exists():
        raise SystemExit(f"Input JSONL does not exist: {input_jsonl}")
    refuse_overwrite(output_jsonl, force, "Output JSONL")

    finetuning = cfg.qwen_finetuning_path()
    if not (finetuning / "prepare_data.py").exists():
        raise SystemExit(f"prepare_data.py not found under {finetuning}. Run vcl setup first.")

    if tokenizer_model_path is None:
        tokenizer = _default_tokenizer_model_path(cfg)
    else:
        tokenizer = str(project_path(tokenizer_model_path))

    ensure_parent(output_jsonl)
    cmd = [
        sys.executable, "prepare_data.py",
        "--device", device or cfg.project.device,
        "--tokenizer_model_path", tokenizer,
        "--input_jsonl", str(input_jsonl),
        "--output_jsonl", str(output_jsonl),
    ]
    _run_or_exit(cmd, cwd=finetuning, what="prepare_data.py")
    print(f"Wrote Qwen training JSONL with audio_codes: {output_jsonl}")
    return output_jsonl


def write_train_metadata(
    output_dir: Path,
    *,
    cfg: Config,
    paths: VoicePaths,
    init_model_path: Path,
    train_jsonl: Path,
    batch_size: int,
    lr: float | str,
    epochs: int,
    speaker: str,
    attn_implementation: str,
) -> Path:
    """Write train_metadata.json provenance next to the checkpoints."""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qwen_repo_commit": run(["git", "rev-parse", "HEAD"], cwd=cfg.qwen_repo_path()),
        "qwen_repo_path": str(cfg.qwen_repo_path()),
        "init_model_path": str(init_model_path),
        "init_model_id": cfg.qwen.init_model_id,
        "train_jsonl": str(train_jsonl),
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "lr": str(lr),
        "epochs": int(epochs),
        "speaker_name": speaker,
        "attn_implementation": attn_implementation,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["torch_cuda_version"] = torch.version.cuda
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["gpu_name"] = props.name
            meta["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:
        meta["torch_error"] = repr(exc)
    try:
        import qwen_tts

        meta["qwen_tts_version"] = getattr(qwen_tts, "__version__", "unknown")
        meta["qwen_tts_file"] = getattr(qwen_tts, "__file__", "unknown")
    except Exception as exc:
        meta["qwen_tts_error"] = repr(exc)

    path = output_dir / "train_metadata.json"
    write_json(path, meta, force=True)
    return path


def train(
    cfg: Config,
    paths: VoicePaths,
    batch_size: int | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    speaker: str | None = None,
    output_dir: Path | None = None,
    attn_implementation: str | None = None,
    allow_download: bool = False,
    force: bool = False,
) -> Path:
    """Fine-tune via upstream sft_12hz.py; returns the output directory."""
    batch_size = batch_size if batch_size is not None else cfg.training.batch_size
    lr = lr if lr is not None else cfg.training.lr
    epochs = epochs if epochs is not None else cfg.training.epochs
    speaker = speaker or paths.speaker
    output_dir = project_path(output_dir) if output_dir else paths.checkpoints_dir

    train_jsonl = paths.train_with_codes_jsonl
    if not train_jsonl.exists():
        raise SystemExit(f"Training JSONL does not exist: {train_jsonl}")
    finetuning = cfg.qwen_finetuning_path()
    if not (finetuning / "sft_12hz.py").exists():
        raise SystemExit(f"sft_12hz.py not found under {finetuning}. Run vcl setup first.")
    init_model = resolve_init_model(cfg, allow_download)
    if attn_implementation and attn_implementation != "auto":
        attn = attn_implementation
    else:
        attn = resolve_attn_implementation(cfg)

    # Overwrite guard: with force, only delete dirs strictly inside outputs/checkpoints.
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"Output directory already contains files: {output_dir}\nRefusing to overwrite. Re-run with --force."
            )
        allowed_root = (ROOT / "outputs" / "checkpoints").resolve()
        resolved = output_dir.resolve()
        if resolved == allowed_root or not resolved.is_relative_to(allowed_root):
            raise SystemExit(f"Refusing to delete a directory outside outputs/checkpoints: {resolved}")
        print(f"Force: removing existing output directory {resolved}")
        shutil.rmtree(resolved)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable, "sft_12hz.py",
        "--init_model_path", str(init_model),
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_jsonl),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(epochs),
        "--speaker_name", speaker,
        "--attn_implementation", attn,
    ]
    command_text = f"cd {shlex.quote(str(finetuning))}\n" + " ".join(shlex.quote(a) for a in argv) + "\n"
    (output_dir / "train_command.txt").write_text(command_text, encoding="utf-8")

    if DEFAULT_CONFIG_PATH.exists():
        shutil.copy(DEFAULT_CONFIG_PATH, output_dir / "voice_config.yaml")
    if paths.dataset_stats.exists():
        shutil.copy(paths.dataset_stats, output_dir / "dataset_stats.json")
    else:
        print(f"WARN: dataset stats not found at {paths.dataset_stats}")

    write_train_metadata(
        output_dir,
        cfg=cfg,
        paths=paths,
        init_model_path=init_model,
        train_jsonl=train_jsonl,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        speaker=speaker,
        attn_implementation=attn,
    )

    log_path = output_dir / "train.log"
    print(f"Starting training (log: {log_path})")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            argv, cwd=finetuning,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:  # stream live and tee to train.log
            print(line, end="")
            log.write(line)
        returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(f"Training failed (exit {returncode}). See log: {log_path}")
    print(f"Training complete. Checkpoints in: {output_dir}")
    return output_dir
