"""Speech generation: fine-tuned single/batch synthesis and zero-shot cloning.

``generate`` is the Content Forge contract (one text -> one WAV),
``generate_batch`` loads the model once for a JSON spec of many lines, and
``zeroshot`` clones a voice from a reference clip with the Base model. Heavy
dependencies (torch, qwen_tts, soundfile) are imported lazily inside functions
so importing this module never needs a GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .config import Config, VoicePaths, project_path
from .utils import ensure_parent, refuse_any_existing, refuse_overwrite, write_json

__all__ = [
    "flash_attn_available",
    "resolve_attn_implementation",
    "load_model",
    "resolve_checkpoint",
    "generate",
    "generate_batch",
    "zeroshot",
]

ATTN_CHOICES = ("auto", "flash_attention_2", "sdpa")


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def resolve_attn_implementation(cfg: Config, no_flash_attn: bool = False) -> str:
    """Resolve cfg.qwen.attn_implementation to a concrete value.

    ``auto`` picks flash_attention_2 when flash-attn is installed, else sdpa.
    ``no_flash_attn`` forces sdpa regardless of the config value.
    """
    if no_flash_attn:
        return "sdpa"
    requested = (cfg.qwen.attn_implementation or "auto").strip()
    if requested == "auto":
        if flash_attn_available():
            return "flash_attention_2"
        print("WARN: flash-attn is not installed; falling back to attn_implementation=sdpa.")
        return "sdpa"
    if requested not in ATTN_CHOICES:
        raise SystemExit(
            f"Invalid qwen.attn_implementation: {requested!r} (expected one of {', '.join(ATTN_CHOICES)})"
        )
    return requested


def load_model(model_path: Path, device: str, attn_implementation: str):
    """Load a Qwen3-TTS model; retry once with sdpa if flash attention fails."""
    import torch
    from qwen_tts import Qwen3TTSModel

    kwargs = {
        "device_map": device,
        "dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }
    try:
        return Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
    except Exception as exc:
        if attn_implementation == "flash_attention_2":
            print(f"WARN: FlashAttention load failed, retrying with sdpa: {exc}")
            kwargs["attn_implementation"] = "sdpa"
            return Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
        raise


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    """Order checkpoints by trailing epoch/step number, then mtime."""
    match = re.search(r"(\d+)(?!.*\d)", path.name)
    number = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (number, mtime)


def resolve_checkpoint(cfg: Config, paths: VoicePaths, explicit: str | None = None) -> Path:
    """Pick the checkpoint dir: explicit arg, config, or newest checkpoint-*."""
    checkpoint: Path | None = None
    if explicit:
        checkpoint = project_path(explicit)
    elif cfg.generation.checkpoint:
        checkpoint = project_path(cfg.generation.checkpoint)
    else:
        base = paths.checkpoints_dir
        candidates = []
        if base.exists():
            candidates = [
                child
                for child in base.iterdir()
                if child.is_dir() and child.name.startswith("checkpoint-") and (child / "config.json").exists()
            ]
        if not candidates:
            raise SystemExit(
                "Could not resolve a fine-tuned checkpoint. Pass an explicit checkpoint, set "
                f"generation.checkpoint in config, or train a model under {base}."
            )
        checkpoint = max(candidates, key=_checkpoint_sort_key)

    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        raise SystemExit(f"Checkpoint does not look complete; missing config.json in {checkpoint}")
    return checkpoint


def _gen_params(cfg: Config, temperature: float | None, top_p: float | None, top_k: int | None):
    """Fill sampling overrides with cfg.generation defaults."""
    return (
        cfg.generation.temperature if temperature is None else temperature,
        cfg.generation.top_p if top_p is None else top_p,
        cfg.generation.top_k if top_k is None else top_k,
    )


def generate(
    cfg: Config,
    paths: VoicePaths,
    text: str,
    out: Path,
    checkpoint: str | None = None,
    speaker: str | None = None,
    language: str | None = None,
    instruct: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    force: bool = False,
    no_flash_attn: bool = False,
) -> Path:
    """Generate one narration WAV with the fine-tuned voice (Content Forge contract)."""
    if not text or not text.strip():
        raise SystemExit("Pass non-empty text to generate.")
    out = project_path(out)
    refuse_overwrite(out, force, "generated WAV")
    ensure_parent(out)

    checkpoint_path = resolve_checkpoint(cfg, paths, checkpoint)
    temperature, top_p, top_k = _gen_params(cfg, temperature, top_p, top_k)
    tts = load_model(checkpoint_path, cfg.project.device, resolve_attn_implementation(cfg, no_flash_attn))

    gen_kwargs = {
        "text": text,
        "speaker": speaker or paths.speaker,
        "language": language or cfg.generation.language,
        "do_sample": True,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }
    if instruct:
        gen_kwargs["instruct"] = instruct

    wavs, sr = tts.generate_custom_voice(**gen_kwargs)
    import soundfile as sf

    sf.write(out, wavs[0], sr)
    print(f"Wrote: {out} (checkpoint={checkpoint_path.name}, sr={sr})")
    return out


_BATCH_OVERRIDE_KEYS = {"checkpoint", "speaker", "language", "temperature", "top_p", "top_k", "no_flash_attn"}


def _load_spec(spec_path: Path) -> list[tuple[str, str]]:
    """Read and validate a batch spec: a non-empty JSON list of {id, text} rows."""
    if not spec_path.exists():
        raise SystemExit(f"Spec file does not exist: {spec_path}")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in spec file {spec_path}: {exc}") from exc
    if not isinstance(spec, list) or not spec:
        raise SystemExit(
            f'Spec must be a non-empty JSON list like [{{"id": "...", "text": "..."}}]: {spec_path}'
        )
    rows = []
    for index, row in enumerate(spec):
        if not isinstance(row, dict) or "id" not in row or "text" not in row:
            raise SystemExit(
                f"Bad spec row {index} in {spec_path}: expected an object with 'id' and 'text', got {row!r}"
            )
        line_id, text = str(row["id"]).strip(), str(row["text"]).strip()
        if not line_id or not text:
            raise SystemExit(f"Bad spec row {index} in {spec_path}: 'id' and 'text' must be non-empty")
        if "/" in line_id or "\\" in line_id:
            raise SystemExit(
                f"Bad spec row {index} in {spec_path}: id {line_id!r} must be a plain filename, not a path"
            )
        rows.append((line_id, text))
    if len({line_id for line_id, _ in rows}) != len(rows):
        raise SystemExit(f"Duplicate ids in spec file {spec_path}")
    return rows


def generate_batch(
    cfg: Config,
    paths: VoicePaths,
    spec_path: Path,
    outdir: Path,
    force: bool = False,
    **gen_overrides,
) -> Path:
    """Generate many clips from a JSON spec with a single model load.

    Writes ``<outdir>/<id>.wav`` per row plus ``<outdir>/manifest.json`` with
    measured durations. All outputs must not exist unless ``force=True``.
    """
    unknown = set(gen_overrides) - _BATCH_OVERRIDE_KEYS
    if unknown:
        raise SystemExit(
            f"Unknown generation overrides: {sorted(unknown)} (allowed: {sorted(_BATCH_OVERRIDE_KEYS)})"
        )
    spec_path = project_path(spec_path)
    outdir = project_path(outdir)
    rows = _load_spec(spec_path)

    targets = [outdir / f"{line_id}.wav" for line_id, _ in rows] + [outdir / "manifest.json"]
    refuse_any_existing(targets, force, "Batch outputs")

    checkpoint = resolve_checkpoint(cfg, paths, gen_overrides.get("checkpoint"))
    temperature, top_p, top_k = _gen_params(
        cfg, gen_overrides.get("temperature"), gen_overrides.get("top_p"), gen_overrides.get("top_k")
    )
    speaker = gen_overrides.get("speaker") or paths.speaker
    language = gen_overrides.get("language") or cfg.generation.language
    no_flash_attn = bool(gen_overrides.get("no_flash_attn", False))

    tts = load_model(checkpoint, cfg.project.device, resolve_attn_implementation(cfg, no_flash_attn))
    import soundfile as sf

    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for line_id, text in rows:
        out = outdir / f"{line_id}.wav"
        wavs, sr = tts.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        sf.write(out, wavs[0], sr)
        duration = round(len(wavs[0]) / sr, 3)
        manifest.append({"id": line_id, "src": f"{line_id}.wav", "duration": duration, "text": text})
        print(f"  {line_id}: {duration:.2f}s -> {out}")

    write_json(outdir / "manifest.json", manifest, force=True)  # overwrite guard already ran above
    print(f"Wrote {len(manifest)} clips + manifest.json to {outdir}")
    return outdir


def _convert_ref_to_wav(src: Path, dst: Path, sample_rate: int) -> Path:
    """Convert any ffmpeg-readable source to mono WAV at the project sample rate."""
    ensure_parent(dst)
    cmd = [
        "ffmpeg", "-nostdin", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dst),
    ]
    result = subprocess.run(cmd)  # stderr surfaces so conversion failures are diagnosable
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed (exit {result.returncode}) converting reference clip: {src}")
    return dst


def zeroshot(
    cfg: Config,
    paths: VoicePaths,
    ref_audio: Path,
    text: str,
    out: Path,
    ref_text: str | None = None,
    language: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    force: bool = False,
    no_flash_attn: bool = False,
) -> Path:
    """Zero-shot clone with the Base model: reference clip + text -> WAV.

    The reference is converted to a ``<out stem>_ref.wav`` next to the output
    (mono, cfg.project.sample_rate), so data/ dataset dirs are never touched.
    """
    if not text or not text.strip():
        raise SystemExit("Pass non-empty text to zeroshot.")
    ref_src = Path(ref_audio).expanduser()
    if not ref_src.exists():
        raise SystemExit(f"Reference clip not found: {ref_src}")
    model_path = cfg.init_model_path()
    if not model_path.is_dir():
        raise SystemExit(
            f"Base model not found: {model_path}\n"
            "Download once with:\n"
            f"  huggingface-cli download {cfg.qwen.init_model_id} --local-dir {model_path}"
        )
    out = project_path(out)
    ref_wav = out.with_name(out.stem + "_ref.wav")
    refuse_any_existing([out, ref_wav], force, "zeroshot outputs")
    if force:  # ffmpeg runs without -y, so pre-clear the targets it writes
        ref_wav.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    ensure_parent(out)

    print(f"Converting reference -> {ref_wav}")
    _convert_ref_to_wav(ref_src, ref_wav, cfg.project.sample_rate)

    x_vector_only = ref_text is None
    mode = "x-vector-only (speaker embedding)" if x_vector_only else "ICL (ref_text provided)"
    print(f"Loading Base model ({mode}) ...")
    tts = load_model(model_path, cfg.project.device, resolve_attn_implementation(cfg, no_flash_attn))

    temperature, top_p, top_k = _gen_params(cfg, temperature, top_p, top_k)
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language=language or cfg.generation.language,
        ref_audio=str(ref_wav),
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    import soundfile as sf

    sf.write(out, wavs[0], sr)
    print(f"Wrote: {out}")
    print(f"Sample rate: {sr}")
    return out
