"""`vcl` command-line interface.

One config source (`config/default.yaml`), per-voice paths derived from
`--speaker`, and every step honors `--force` overwrite safety. Commands:

    vcl setup       clone + pin + patch the vendored Qwen3-TTS repo
    vcl check       report system readiness
    vcl run         extract → clean → split → transcribe → dataset
    vcl extract     pull mono 24 kHz audio out of a recording (ffmpeg)
    vcl clean       conservatively trim long silence
    vcl split       cut speech into 3-12 s chunks
    vcl transcribe  ASR the chunks (--review / --apply-review for proofreading)
    vcl dataset     pick a reference clip and build train_raw.jsonl
    vcl prepare     add Qwen audio codes (train_with_codes.jsonl)
    vcl train       fine-tune via upstream sft_12hz.py
    vcl generate    text → WAV (fine-tuned checkpoint, batch spec, or zero-shot)
    vcl ui          launch the Gradio web UI
    vcl migrate     move a legacy data/ layout into data/voices/<speaker>/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import Config, VoicePaths, project_path


def _add_common(parser: argparse.ArgumentParser, force: bool = True) -> None:
    parser.add_argument("--speaker", default=None, help="Voice name (default: project.speaker in config)")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: config/default.yaml)")
    if force:
        parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")


def _load(args) -> tuple[Config, VoicePaths]:
    cfg = Config.load(args.config)
    paths = cfg.paths_for(args.speaker or cfg.project.speaker)
    return cfg, paths


def cmd_setup(args) -> int:
    from .train import resolve_init_model, setup_vendor

    cfg, _ = _load(args)
    setup_vendor(cfg)
    if args.download:
        resolve_init_model(cfg, allow_download=True)
    print("Setup complete.")
    return 0


def cmd_check(args) -> int:
    from .system import check_system, format_report

    cfg, _ = _load(args)
    results = check_system(cfg, min_free_gb=args.min_free_gb)
    print(format_report(results))
    return 1 if any(r.status == "FAIL" for r in results) else 0


def _resolve_source(input_value: str, paths: VoicePaths, force: bool) -> Path:
    """Turn --input into a local audio file, downloading first when it's a URL."""
    from .sources import download_source, is_url

    if is_url(input_value):
        return download_source(input_value, paths.raw_dir, force=force)
    return project_path(input_value)


def cmd_extract(args) -> int:
    from .audio import extract_audio

    cfg, paths = _load(args)
    output = project_path(args.output) if args.output else paths.extracted_audio
    source = _resolve_source(args.input, paths, args.force)
    extract_audio(source, output, args.sample_rate or cfg.project.sample_rate, force=args.force)
    print(f"Wrote: {output}")
    return 0


def cmd_clean(args) -> int:
    from .audio import clean_audio

    cfg, paths = _load(args)
    input_path = project_path(args.input) if args.input else paths.extracted_audio
    output = project_path(args.output) if args.output else paths.cleaned_audio
    clean_audio(input_path, output, cfg.audio, cfg.project.sample_rate, denoise=args.denoise, force=args.force)
    return 0


def cmd_split(args) -> int:
    from .audio import split_audio

    cfg, paths = _load(args)
    input_path = project_path(args.input) if args.input else paths.cleaned_audio
    split_audio(input_path, paths, cfg.audio, cfg.project.sample_rate, force=args.force)
    return 0


def cmd_transcribe(args) -> int:
    from .transcribe import apply_review, export_review, transcribe_chunks

    cfg, paths = _load(args)
    if args.apply_review:
        n = apply_review(paths, force=args.force)
        print(f"Applied {n} reviewed rows to {paths.transcripts_jsonl}")
        return 0
    if args.review:
        out = export_review(paths, force=args.force)
        print(f"Wrote review TSV: {out}\nEdit it, then run: vcl transcribe --apply-review")
        return 0
    for key in ("backend", "model", "device", "compute_type", "language"):
        value = getattr(args, f"asr_{key}")
        if value is not None:
            setattr(cfg.asr, key, value)
    rows = transcribe_chunks(paths, cfg.asr, force=args.force)
    print(f"Wrote {len(rows)} transcripts to {paths.transcripts_jsonl}")
    print("Proofread them: vcl transcribe --review")
    return 0


def cmd_dataset(args) -> int:
    from .dataset import build_dataset

    cfg, paths = _load(args)
    stats = build_dataset(
        paths,
        cfg.audio,
        cfg.project.sample_rate,
        min_minutes=args.min_minutes,
        source_ref=project_path(args.source_ref) if args.source_ref else None,
        path_mode=args.path_mode,
        force=args.force,
    )
    print(f"Wrote {stats['clips']} clips ({stats['total_minutes']:.1f} min) to {paths.train_raw_jsonl}")
    print(f"Reference: {stats['reference_audio']}")
    return 0


def cmd_prepare(args) -> int:
    from .train import prepare_codes

    cfg, paths = _load(args)
    out = prepare_codes(cfg, paths, device=args.device, force=args.force)
    print(f"Wrote: {out}")
    return 0


def cmd_train(args) -> int:
    from .train import train

    cfg, paths = _load(args)
    output_dir = train(
        cfg,
        paths,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=project_path(args.output_dir) if args.output_dir else None,
        attn_implementation=args.attn_implementation,
        allow_download=args.allow_download,
        force=args.force,
    )
    print(f"Checkpoints written under: {output_dir}")
    return 0


def cmd_generate(args) -> int:
    from .generate import generate, generate_batch, zeroshot

    cfg, paths = _load(args)
    if args.device:
        cfg.project.device = args.device

    gen_overrides = {
        "checkpoint": args.checkpoint,
        "speaker": args.speaker_name,
        "language": args.language,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_flash_attn": args.no_flash_attn,
    }

    if args.spec:
        if not args.outdir:
            raise SystemExit("--spec requires --outdir")
        manifest = generate_batch(
            cfg, paths, project_path(args.spec), project_path(args.outdir), force=args.force, **gen_overrides
        )
        print(f"Wrote batch manifest: {manifest}")
        return 0

    text = args.text
    if args.text_file:
        text = project_path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Pass --text, --text_file, or --spec.")
    out = project_path(args.out) if args.out else paths.generated_dir / "test.wav"

    if args.zeroshot:
        if not args.ref:
            raise SystemExit("--zeroshot requires --ref REF_AUDIO")
        result = zeroshot(
            cfg, paths, project_path(args.ref), text, out, ref_text=args.ref_text,
            language=args.language, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, force=args.force, no_flash_attn=args.no_flash_attn,
        )
    else:
        result = generate(cfg, paths, text, out, instruct=args.instruct, force=args.force, **gen_overrides)
    print(f"Wrote: {result}")
    return 0


def cmd_run(args) -> int:
    cfg, paths = _load(args)
    from .audio import clean_audio, extract_audio, split_audio
    from .dataset import build_dataset
    from .transcribe import transcribe_chunks

    source = _resolve_source(args.input, paths, args.force)
    extract_audio(source, paths.extracted_audio, cfg.project.sample_rate, force=args.force)
    print(f"[1/5] extracted -> {paths.extracted_audio}")
    stats = clean_audio(paths.extracted_audio, paths.cleaned_audio, cfg.audio, cfg.project.sample_rate,
                        denoise=args.denoise, force=args.force)
    print(f"[2/5] cleaned -> {paths.cleaned_audio} "
          f"({stats['before']['duration_seconds']:.1f}s -> {stats['after']['duration_seconds']:.1f}s)")
    metadata = split_audio(paths.cleaned_audio, paths, cfg.audio, cfg.project.sample_rate, force=args.force)
    print(f"[3/5] split -> {len(metadata['clips'])} chunks (rejected {len(metadata['rejected'])})")
    rows = transcribe_chunks(paths, cfg.asr, force=args.force)
    print(f"[4/5] transcribed {len(rows)} chunks")
    stats = build_dataset(paths, cfg.audio, cfg.project.sample_rate, min_minutes=args.min_minutes, force=args.force)
    print(f"[5/5] dataset -> {paths.train_raw_jsonl} ({stats['clips']} clips, {stats['total_minutes']:.1f} min)")
    print("Next: proofread with 'vcl transcribe --review', then 'vcl prepare' and 'vcl train'.")
    return 0


_LEGACY_DATA_DIRS = ("raw", "extracted", "cleaned", "chunks", "transcripts", "dataset", "reference")


def cmd_migrate(args) -> int:
    from .config import ROOT

    cfg, paths = _load(args)
    moves: list[tuple[Path, Path]] = []
    for name in _LEGACY_DATA_DIRS:
        src = ROOT / "data" / name
        if not src.exists():
            continue
        dst = paths.raw_dir if name == "raw" else paths.voice_dir / name
        moves.append((src, dst))
    if not moves:
        print("No legacy data/ directories found; nothing to migrate.")
        return 0
    for src, dst in moves:
        print(f"  {src} -> {dst}")
    if args.dry_run:
        print("Dry run; no files moved.")
        return 0
    conflicts = [dst for _, dst in moves if dst.exists() and any(dst.iterdir())]
    if conflicts and not args.force:
        shown = "\n".join(f"  {c}" for c in conflicts)
        raise SystemExit(f"Target directories are not empty:\n{shown}\nRefusing to overwrite. Re-run with --force to merge.")
    for src, dst in moves:
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.move(str(item), str(dst / item.name))
        src.rmdir()
    print(f"Migrated legacy data/ into {paths.voice_dir}")
    print("Note: old metadata/transcripts contain absolute paths; readers still resolve them.")
    return 0


def cmd_ui(args) -> int:
    cfg, _ = _load(args)
    try:
        from .ui import APP_CSS, app_theme, build_app
    except ImportError as exc:
        raise SystemExit(f"Gradio UI unavailable ({exc}). Install with: pip install 'voice-clone-lab[ui]'") from exc
    app = build_app(cfg)
    # Gradio 6 auto-scopes the css= parameter under .gradio-container, which breaks
    # :root/container-level rules — deliver the stylesheet raw via head= instead.
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               theme=app_theme(), head=f"<style>{APP_CSS}</style>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vcl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="Clone/pin/patch the vendored Qwen3-TTS repo")
    _add_common(p, force=False)
    p.add_argument("--download", action="store_true", help="Also download the base model weights")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("check", help="Report system readiness")
    _add_common(p, force=False)
    p.add_argument("--min-free-gb", type=int, default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="Extract → clean → split → transcribe → dataset")
    _add_common(p)
    p.add_argument("--input", required=True, help="Raw recording (wav/mp4/m4a/...) or an http(s) URL (YouTube etc.)")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--min-minutes", type=float, default=5.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("extract", help="Extract mono WAV from a recording")
    _add_common(p)
    p.add_argument("--input", required=True, help="Audio/video file or an http(s) URL (YouTube etc.)")
    p.add_argument("--output", default=None)
    p.add_argument("--sample-rate", type=int, default=None)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("clean", help="Trim long silence from extracted audio")
    _add_common(p)
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--denoise", action="store_true")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("split", help="Split cleaned audio into chunks")
    _add_common(p)
    p.add_argument("--input", default=None)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("transcribe", help="Transcribe chunks locally (or manage the review TSV)")
    _add_common(p)
    p.add_argument("--review", action="store_true", help="Export an editable review TSV")
    p.add_argument("--apply-review", action="store_true", help="Apply the edited review TSV")
    p.add_argument("--asr-backend", default=None)
    p.add_argument("--asr-model", default=None)
    p.add_argument("--asr-device", default=None)
    p.add_argument("--asr-compute-type", default=None)
    p.add_argument("--asr-language", default=None)
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("dataset", help="Build train_raw.jsonl and pick a reference clip")
    _add_common(p)
    p.add_argument("--min-minutes", type=float, default=5.0)
    p.add_argument("--source-ref", default=None, help="Use this clip as the reference audio")
    p.add_argument("--path-mode", default="qwen-relative", choices=["qwen-relative", "project-relative", "absolute"])
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("prepare", help="Add Qwen audio codes to the dataset")
    _add_common(p)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Fine-tune Qwen3-TTS on the prepared dataset")
    _add_common(p)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--attn-implementation", default=None, choices=["auto", "flash_attention_2", "sdpa"])
    p.add_argument("--allow-download", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="Text → WAV with a fine-tuned checkpoint (or zero-shot)")
    _add_common(p)
    p.add_argument("--text", default=None)
    p.add_argument("--text-file", default=None)
    p.add_argument("--out", default=None, help="Output WAV (default: outputs/generated/<speaker>/test.wav)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--speaker-name", default=None, help="Speaker name baked into the checkpoint at train time")
    p.add_argument("--language", default=None)
    p.add_argument("--instruct", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", default=None, help="Override generation device, e.g. cuda:1")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--spec", default=None, help="Batch mode: JSON list of {id, text}")
    p.add_argument("--outdir", default=None, help="Batch mode output directory")
    p.add_argument("--zeroshot", action="store_true", help="Use the base model with --ref audio (no training)")
    p.add_argument("--ref", default=None, help="Reference audio for --zeroshot")
    p.add_argument("--ref-text", default=None, help="Transcript of --ref (improves zero-shot quality)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("ui", help="Launch the Gradio web UI")
    _add_common(p, force=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("migrate", help="Move legacy data/* layout into data/voices/<speaker>/")
    _add_common(p)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
