"""Gradio web UI for the voice-clone pipeline (``vcl ui``).

Thin wiring over the same package functions the CLI calls — no logic forks.
All event handlers are module-level functions with explicit arguments so they
are testable without a browser; :func:`build_app` only assembles components.
Pipeline steps print progress, so handlers capture stdout into log text boxes
(:func:`_run_logged`) and turn ``SystemExit`` into ``gr.Error`` — never a bare
traceback.

Training runs in a daemon thread (``prepare_codes`` → ``train``) whose stdout
is NOT redirected: ``train.train`` streams the sft_12hz.py subprocess to the
console and tees it to ``outputs/checkpoints/<speaker>/train.log``, which the
UI tails with a ``gr.Timer``. Cancel is a ``threading.Event`` checked between
the prepare and train phases — sft_12hz.py has no cancel hook, so a training
subprocess that already started runs to completion.
"""

from __future__ import annotations

import contextlib
import csv
import io
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr

from . import config as _config
from .config import Config
from .utils import ensure_parent, read_jsonl

__all__ = [
    "build_app",
    "app_theme",
    "APP_CSS",
    "list_voices",
    "refresh_voices",
    "handle_system_check",
    "handle_setup",
    "handle_run_pipeline",
    "handle_transcribe",
    "load_transcripts",
    "load_transcripts_silent",
    "refresh_transcript_table",
    "save_transcripts",
    "handle_build_dataset",
    "handle_train_start",
    "handle_train_cancel",
    "train_status",
    "train_log_tail",
    "poll_train",
    "list_checkpoints",
    "refresh_checkpoints",
    "handle_generate_finetuned",
    "handle_generate_zeroshot",
    "handle_generate",
]

ATTN_CHOICES = ["auto", "flash_attention_2", "sdpa"]
LOG_TAIL_CHARS = 8000

# Module-level training-run state: at most one run at a time (see Tab 3).
_train_lock = threading.Lock()
_train_state: dict[str, Any] = {"thread": None, "cancel": threading.Event(), "status": "idle"}


# --- helpers -----------------------------------------------------------------


def _run_logged(fn, *args, **kwargs) -> tuple[str, Any]:
    """Run a pipeline step, capturing its printed progress.

    Returns ``(captured_stdout, result)``. A step's ``SystemExit`` (the
    codebase's user-facing error channel) becomes ``gr.Error`` instead of a
    traceback.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except SystemExit as exc:
        raise gr.Error(str(exc)) from exc
    return buf.getvalue(), result


def _require_speaker(speaker: str | None) -> str:
    speaker = (speaker or "").strip()
    if not speaker:
        raise gr.Error("Pick or type a speaker name first.")
    if "/" in speaker or "\\" in speaker:
        raise gr.Error(f"Speaker name must be a plain directory name, not a path: {speaker!r}")
    return speaker


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_table(table) -> list[list[str]]:
    """Dataframe value -> rows of ``[audio, text]`` strings (None cells -> '')."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):  # tolerate a pandas DataFrame
        table = table.values.tolist()
    rows = []
    for row in table:
        audio, text = (list(row) + ["", ""])[:2]
        rows.append(["" if audio is None else str(audio), "" if text is None else str(text)])
    return rows


# --- speaker / checkpoint listing ---------------------------------------------


def list_voices(cfg: Config) -> list[str]:
    """Existing voice names = subdirs of ``data/voices/``, plus the configured speaker."""
    voices_dir = _config.ROOT / "data" / "voices"
    names = sorted(p.name for p in voices_dir.iterdir() if p.is_dir()) if voices_dir.is_dir() else []
    if cfg.project.speaker and cfg.project.speaker not in names:
        names.insert(0, cfg.project.speaker)
    return names


def refresh_voices(cfg: Config, current: str | None) -> dict:
    """Update the speaker Dropdown, keeping the typed-in value when set."""
    choices = list_voices(cfg)
    current = (current or "").strip()
    value = current or (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def list_checkpoints(cfg: Config, speaker: str | None) -> list[str]:
    """Sorted ``checkpoint-*`` dirs under ``outputs/checkpoints/<speaker>`` (oldest first)."""
    from .generate import _checkpoint_sort_key  # same ordering resolve_checkpoint uses

    speaker = (speaker or "").strip()
    if not speaker:
        return []
    base = cfg.paths_for(speaker).checkpoints_dir
    if not base.is_dir():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    return [str(d) for d in sorted(dirs, key=_checkpoint_sort_key)]


def refresh_checkpoints(cfg: Config, speaker: str | None) -> dict:
    """Update the checkpoint Dropdown, preselecting the newest checkpoint."""
    choices = list_checkpoints(cfg, speaker)
    return gr.update(choices=choices, value=choices[-1] if choices else None)


# --- Tab 1: Setup --------------------------------------------------------------


def handle_system_check(cfg: Config) -> str:
    """Run ``check_system`` and render ``format_report`` for the report box."""
    from .system import check_system, format_report

    log, results = _run_logged(check_system, cfg)
    return log + format_report(results)


def handle_setup(cfg: Config, download_model: bool) -> str:
    """Run ``setup_vendor``; optionally also download the base model."""
    from .train import resolve_init_model, setup_vendor

    log, _ = _run_logged(setup_vendor, cfg)
    if download_model:
        download_log, _ = _run_logged(resolve_init_model, cfg, allow_download=True)
        log += download_log
    return log + "Setup complete."


# --- Tab 2: Data ---------------------------------------------------------------


def handle_run_pipeline(cfg: Config, speaker: str | None, source_path: str | None, source_url: str | None) -> str:
    """extract -> clean -> split on an uploaded/recorded file or a URL (mirrors `vcl run`)."""
    from .audio import clean_audio, extract_audio, split_audio
    from .sources import download_source, is_url

    paths = cfg.paths_for(_require_speaker(speaker))
    log = ""
    if source_path:
        source = Path(source_path)
        if not source.exists():
            raise gr.Error(f"Audio file does not exist: {source}")
    elif is_url(source_url):
        dl_log, source = _run_logged(download_source, source_url, paths.raw_dir)
        log += dl_log
    else:
        raise gr.Error("Upload or record an audio file, or paste a YouTube/URL source.")

    step_log, _ = _run_logged(extract_audio, source, paths.extracted_audio, cfg.project.sample_rate, force=True)
    log += step_log
    clean_log, stats = _run_logged(
        clean_audio, paths.extracted_audio, paths.cleaned_audio, cfg.audio, cfg.project.sample_rate, force=True
    )
    log += clean_log
    split_log, metadata = _run_logged(
        split_audio, paths.cleaned_audio, paths, cfg.audio, cfg.project.sample_rate, force=True
    )
    log += split_log
    return (
        f"{log}"
        f"Pipeline done: extracted -> {paths.extracted_audio}\n"
        f"cleaned ({stats['before']['duration_seconds']:.1f}s -> {stats['after']['duration_seconds']:.1f}s)\n"
        f"split -> {len(metadata['clips'])} chunks (rejected {len(metadata['rejected'])})\n"
        "Next: Transcribe (ASR), proofread the table, then Build dataset."
    )


def handle_transcribe(cfg: Config, speaker: str | None) -> str:
    """ASR the split chunks into transcripts.jsonl (mirrors `vcl transcribe`)."""
    from .transcribe import transcribe_chunks

    paths = cfg.paths_for(_require_speaker(speaker))
    has_chunks = paths.chunk_metadata.exists() or any(paths.chunks_dir.glob("chunk_*.wav"))
    if not has_chunks:
        raise gr.Error(f"No chunks found under {paths.chunks_dir} — run the audio pipeline first.")
    log, rows = _run_logged(transcribe_chunks, paths, cfg.asr, force=True)
    return f"{log}Transcribed {len(rows)} chunks. Proofread the table, then Save transcripts."


def load_transcripts(cfg: Config, speaker: str | None) -> list[list[str]]:
    """Load transcripts.jsonl as ``[[audio, text], ...]`` rows for the Dataframe."""
    paths = cfg.paths_for(_require_speaker(speaker))
    if not paths.transcripts_jsonl.exists():
        raise gr.Error(f"No transcripts yet ({paths.transcripts_jsonl}) — run Transcribe (ASR) first.")
    rows = read_jsonl(paths.transcripts_jsonl)
    return [[str(row.get("audio", "")), str(row.get("text", ""))] for row in rows]


def load_transcripts_silent(cfg: Config, speaker: str | None) -> list[list[str]]:
    """Like :func:`load_transcripts` but returns ``[]`` instead of raising.

    Used on speaker change so switching voices refreshes the table (or clears
    it for a voice with no transcripts) without an error popup.
    """
    speaker = (speaker or "").strip()
    if not speaker or "/" in speaker or "\\" in speaker:
        return []
    paths = cfg.paths_for(speaker)
    if not paths.transcripts_jsonl.exists():
        return []
    rows = read_jsonl(paths.transcripts_jsonl)
    return [[str(row.get("audio", "")), str(row.get("text", ""))] for row in rows]


def refresh_transcript_table(cfg: Config, speaker: str | None) -> dict:
    """Speaker-change wiring: reload the Dataframe, clearing it for voices with no transcripts.

    A bare ``[]`` return is collapsed to a no-op by Gradio's postprocessing, so
    wrap it in an explicit ``gr.update``.
    """
    return gr.update(value=load_transcripts_silent(cfg, speaker))


def save_transcripts(cfg: Config, speaker: str | None, table) -> str:
    """Write edited rows through the CLI's review path: review TSV -> apply_review."""
    from .transcribe import apply_review

    paths = cfg.paths_for(_require_speaker(speaker))
    rows = _normalize_table(table)
    if not rows:
        raise gr.Error("Transcript table is empty — load transcripts first.")

    # Guard against saving stale rows from a previously selected voice into
    # this speaker's transcripts (the table persists across speaker changes).
    marker = "data/voices/"
    foreign = [
        audio
        for audio, _ in rows
        if marker in audio.replace("\\", "/") and f"{marker}{paths.speaker}/" not in audio.replace("\\", "/")
    ]
    if foreign:
        raise gr.Error(
            f"The table contains rows for a different voice ({foreign[0]}). "
            "Load transcripts for the current speaker (or run Transcribe) before saving."
        )

    # Same `audio\ttext` format export_review writes, so apply_review reads it back.
    ensure_parent(paths.review_tsv)
    with paths.review_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio", "text"])
        writer.writerows(rows)

    log, n = _run_logged(apply_review, paths, force=True)
    return f"{log}Saved {n} reviewed rows to {paths.transcripts_jsonl}"


def handle_build_dataset(cfg: Config, speaker: str | None) -> str:
    """Build train_raw.jsonl + pick a reference clip (mirrors `vcl dataset`)."""
    from .dataset import build_dataset

    paths = cfg.paths_for(_require_speaker(speaker))
    if not paths.transcripts_jsonl.exists():
        raise gr.Error(f"No transcripts yet ({paths.transcripts_jsonl}) — run Transcribe (ASR) first.")
    log, stats = _run_logged(build_dataset, paths, cfg.audio, cfg.project.sample_rate, min_minutes=5.0, force=True)
    return (
        f"{log}"
        f"Dataset ready: {stats['clips']} clips, {stats['total_minutes']:.1f} min\n"
        f"Reference: {stats['reference_audio']}\n"
        "Next: the Train tab."
    )


# --- Tab 3: Train --------------------------------------------------------------


def _train_worker(
    cfg: Config,
    paths,
    batch_size: int,
    lr: float,
    epochs: int,
    attn_implementation: str | None,
    cancel: threading.Event,
) -> None:
    """Daemon-thread body: prepare_codes, then train (stdout not redirected).

    ``train.train`` tees sft_12hz.py output to train.log, which the UI tails.
    """
    from .train import prepare_codes, train

    try:
        prepare_codes(cfg, paths, force=True)
        if cancel.is_set():
            _train_state["status"] = "cancelled before training started"
            print("UI: cancel requested; sft_12hz.py was never launched.")
            return
        train(
            cfg,
            paths,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            attn_implementation=attn_implementation,
            force=True,
        )
        _train_state["status"] = "done"
    except SystemExit as exc:
        _train_state["status"] = f"failed: {exc}"
    except Exception as exc:  # never let the daemon thread die silently
        _train_state["status"] = f"error: {exc.__class__.__name__}: {exc}"


def handle_train_start(
    cfg: Config,
    speaker: str | None,
    batch_size,
    lr,
    epochs,
    attn_implementation: str | None,
    overwrite: bool,
) -> tuple[str, dict]:
    """Launch prepare+train in a daemon thread; disables Start while active.

    Retraining a speaker REPLACES its previous checkpoints — only when the
    user explicitly ticked Overwrite; otherwise refuse with a clear message.
    """
    paths = cfg.paths_for(_require_speaker(speaker))
    batch_size, lr, epochs = int(batch_size), float(lr), int(epochs)
    attn_implementation = attn_implementation or None
    has_checkpoints = paths.checkpoints_dir.is_dir() and any(
        d.is_dir() and d.name.startswith("checkpoint-") for d in paths.checkpoints_dir.iterdir()
    )
    if has_checkpoints and not overwrite:
        gr.Warning(
            f"'{paths.speaker}' already has trained checkpoints. Retraining REPLACES them "
            "(the old ones are deleted). Tick 'Overwrite previous checkpoints' and Start again, "
            "or pick a new speaker name to keep both."
        )
        return "Not started — see the warning.", gr.update(interactive=True)
    with _train_lock:
        thread = _train_state.get("thread")
        if thread is not None and thread.is_alive():
            gr.Warning("A training run is already active.")
            return "A training run is already active.", gr.update(interactive=False)
        cancel = threading.Event()
        worker = threading.Thread(
            target=_train_worker,
            args=(cfg, paths, batch_size, lr, epochs, attn_implementation, cancel),
            daemon=True,
        )
        _train_state.update(thread=worker, cancel=cancel, status="running (prepare_data.py)")
        worker.start()
    return (
        f"Training started for '{paths.speaker}': prepare_codes -> train "
        f"(batch_size={batch_size}, lr={lr:g}, epochs={epochs}, attn={attn_implementation or 'auto'}).\n"
        f"Log tailed from: {paths.checkpoints_dir / 'train.log'}",
        gr.update(interactive=False),
    )


def handle_train_cancel() -> str:
    """Request cancellation of the active run (phase-boundary granularity)."""
    with _train_lock:
        thread = _train_state.get("thread")
        if thread is None or not thread.is_alive():
            return "No training run is active."
        _train_state["cancel"].set()
        _train_state["status"] = "cancel requested"
    return (
        "Cancel requested. It takes effect between the prepare and train phases; "
        "an sft_12hz.py run that already started has no cancel hook and runs to completion "
        "(kill it from the console if you must stop it sooner)."
    )


def train_status() -> str:
    """Current training-run status line."""
    with _train_lock:
        thread = _train_state.get("thread")
        active = thread is not None and thread.is_alive()
        status = _train_state["status"]
    return f"{status} (active)" if active else str(status)


def train_log_tail(cfg: Config, speaker: str | None, max_chars: int = LOG_TAIL_CHARS) -> str:
    """Tail of ``outputs/checkpoints/<speaker>/train.log`` ('' when absent)."""
    speaker = (speaker or "").strip()
    if not speaker:
        return ""
    log_path = cfg.paths_for(speaker).checkpoints_dir / "train.log"
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_chars * 2))  # byte margin for UTF-8
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-max_chars:]


def poll_train(cfg: Config, speaker: str | None) -> tuple[str, str, dict, dict]:
    """gr.Timer tick: status, train.log tail, Start interactivity, checkpoint list.

    The checkpoint refresh keeps the Generate tab current after a run finishes
    (the dropdown otherwise only updates on speaker change or manual refresh).
    """
    with _train_lock:
        thread = _train_state.get("thread")
        active = thread is not None and thread.is_alive()
    return (
        train_status(),
        train_log_tail(cfg, speaker),
        gr.update(interactive=not active),
        refresh_checkpoints(cfg, speaker),
    )


# --- Tab 4: Generate -----------------------------------------------------------


def handle_generate_finetuned(
    cfg: Config,
    speaker: str | None,
    checkpoint: str | None,
    text: str | None,
    instruct: str | None,
    temperature,
    top_p,
    top_k,
    language: str | None,
) -> str:
    """Generate one WAV with the selected fine-tuned checkpoint (mirrors `vcl generate`)."""
    from .generate import generate

    paths = cfg.paths_for(_require_speaker(speaker))
    if not text or not str(text).strip():
        raise gr.Error("Enter text to synthesize.")
    out = paths.generated_dir / f"ui_{_timestamp()}.wav"
    _run_logged(
        generate,
        cfg,
        paths,
        text,
        out,
        checkpoint=checkpoint or None,
        instruct=str(instruct).strip() or None if instruct else None,
        language=str(language).strip() or None if language else None,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        force=True,
    )
    return str(out)


def handle_generate_zeroshot(
    cfg: Config,
    speaker: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    text: str | None,
    temperature,
    top_p,
    top_k,
    language: str | None,
) -> str:
    """Zero-shot clone from a reference clip with the Base model (mirrors `vcl generate --zeroshot`)."""
    from .generate import zeroshot

    paths = cfg.paths_for(_require_speaker(speaker))
    if not ref_audio:
        raise gr.Error("Upload a reference audio clip for zero-shot generation.")
    if not text or not str(text).strip():
        raise gr.Error("Enter text to synthesize.")
    out = paths.generated_dir / f"ui_{_timestamp()}.wav"
    _run_logged(
        zeroshot,
        cfg,
        paths,
        Path(ref_audio),
        text,
        out,
        ref_text=str(ref_text).strip() or None if ref_text else None,
        language=str(language).strip() or None if language else None,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        force=True,
    )
    return str(out)


def handle_generate(
    cfg: Config,
    speaker: str | None,
    mode: str,
    checkpoint: str | None,
    text: str | None,
    instruct: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    temperature,
    top_p,
    top_k,
    language: str | None,
) -> str:
    """Dispatch the Generate button to the fine-tuned or zero-shot handler."""
    if mode == "zero-shot":
        return handle_generate_zeroshot(cfg, speaker, ref_audio, ref_text, text, temperature, top_p, top_k, language)
    return handle_generate_finetuned(cfg, speaker, checkpoint, text, instruct, temperature, top_p, top_k, language)


def _mode_groups(mode: str) -> tuple[dict, dict]:
    """Show only the input group matching the generation mode."""
    return gr.update(visible=mode != "zero-shot"), gr.update(visible=mode == "zero-shot")


# --- app assembly --------------------------------------------------------------
#
# Presentation only: theme, layout, and styling live here. Handlers above are
# untouched by restyling; wiring (inputs/outputs/fns) stays identical.

_HEADER_HTML = """
<div class="vcl-hero">
  <div class="vcl-hero-title">Voice Clone Lab</div>
  <div class="vcl-hero-sub">Fine-tune Qwen3-TTS on your own voice — same pipeline as the <code>vcl</code> CLI.</div>
  <div class="vcl-hero-consent">Clone only your own voice or voices you have written permission for.</div>
</div>
"""

_FOOTER_HTML = """
<div class="vcl-footer">
  Tip: every button calls the same package functions as the CLI — CLI equivalent: <code>vcl &lt;command&gt;</code>
  (<code>vcl run</code>, <code>vcl train</code>, <code>vcl generate</code>, ...).
</div>
"""

_CANCEL_NOTE_HTML = """
<div class="vcl-note">
  Runs <code>prepare_codes</code> then <code>train</code> in a background thread. Cancel sets a flag
  checked between the two phases — a running <code>sft_12hz.py</code> has no cancel hook and runs
  to completion.
</div>
"""


def _step_html(number: int, title: str) -> str:
    """Numbered step chip used as section headers on the Data tab."""
    return f'<div class="vcl-step"><span class="vcl-step-num">{number}</span>{title}</div>'


# Offline-safe system font stacks (no external font downloads).
_FONT_STACK = ["system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
_FONT_MONO_STACK = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", "monospace"]


def app_theme() -> gr.themes.Soft:
    """Base theme (indigo/slate); the dark-studio look itself is forced via APP_CSS."""
    return gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=_FONT_STACK,
        font_mono=_FONT_MONO_STACK,
    )


APP_CSS = """
/* ---- dark studio palette (forced for both light and dark OS preference) ---- */
:root, .dark {
  --vcl-bg: #0a0d13;
  --vcl-panel: #111722;
  --vcl-panel-2: #161e2b;
  --vcl-border: rgba(148, 163, 184, 0.14);
  --vcl-text: #e8edf5;
  --vcl-dim: #93a1b5;
  --vcl-accent: #7c6cf6;
  --vcl-accent-2: #a78bfa;
  --vcl-glow: rgba(124, 108, 246, 0.35);

  --body-background-fill: var(--vcl-bg);
  --background-fill-primary: var(--vcl-panel);
  --background-fill-secondary: var(--vcl-panel-2);
  --block-background-fill: var(--vcl-panel);
  --block-border-color: var(--vcl-border);
  --block-title-text-color: var(--vcl-dim);
  --body-text-color: var(--vcl-text);
  --body-text-color-subdued: var(--vcl-dim);
  --input-background-fill: #0c1119;
  --border-color-primary: var(--vcl-border);
  --border-color-accent: var(--vcl-accent);
  --button-secondary-background-fill: var(--vcl-panel-2);
  --button-secondary-text-color: var(--vcl-text);
  --color-accent: var(--vcl-accent);
  --link-text-color: var(--vcl-accent-2);
  --stat-background-fill: var(--vcl-panel-2);
  --table-even-background-fill: var(--vcl-panel-2);
  --table-odd-background-fill: var(--vcl-panel);
  --loader-color: var(--vcl-accent);
}

.gradio-container {
  width: 100% !important;
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  background:
    radial-gradient(1200px 420px at 18% -10%, rgba(124, 108, 246, 0.13), transparent 60%),
    radial-gradient(900px 320px at 85% -10%, rgba(167, 139, 250, 0.09), transparent 60%),
    var(--vcl-bg) !important;
  padding-top: 18px !important;
}

/* ultrawide: give the app more room instead of a narrow strip */
@media (min-width: 1800px) {
  .gradio-container { max-width: 1440px !important; }
}
@media (min-width: 2600px) {
  .gradio-container { max-width: 1700px !important; }
}

/* ---- hero band ---- */
.vcl-hero {
  border: 1px solid rgba(124, 108, 246, 0.35);
  border-radius: 16px;
  padding: 26px 30px;
  margin-bottom: 10px;
  background: linear-gradient(135deg, rgba(124, 108, 246, 0.30), rgba(167, 139, 250, 0.08) 55%, transparent),
              var(--vcl-panel);
  box-shadow: 0 0 24px rgba(124, 108, 246, 0.15), 0 12px 40px rgba(0, 0, 0, 0.45);
}
.vcl-hero-title { font-size: 1.75em; font-weight: 750; letter-spacing: -0.02em; color: var(--vcl-text); }
.vcl-hero-sub { margin-top: 4px; color: var(--vcl-dim); font-size: 0.98em; }
.vcl-hero-sub code, .vcl-footer code, .vcl-note code { color: var(--vcl-accent-2); }
.vcl-hero-consent { margin-top: 10px; font-size: 0.82em; color: var(--vcl-dim); opacity: 0.85; }

/* ---- pill tab navigation (Gradio 6: .tab-container) ---- */
.tab-wrapper { border-bottom: none !important; box-shadow: none !important; }
.tab-container { border: none !important; box-shadow: none !important; gap: 8px !important; margin: 6px 0 4px; }
.tab-container button {
  border-radius: 999px !important;
  border: 1px solid var(--vcl-border) !important;
  background: transparent !important;
  color: var(--vcl-dim) !important;
  padding: 7px 18px !important;
  font-weight: 600 !important;
}
.tab-container button:hover { color: var(--vcl-text) !important; border-color: rgba(148, 163, 184, 0.35) !important; }
.tab-container button.selected {
  background: rgba(124, 108, 246, 0.16) !important;
  border-color: var(--vcl-accent) !important;
  color: var(--vcl-text) !important;
  box-shadow: 0 0 16px var(--vcl-glow);
}

/* ---- form labels: plain dim text, no chips ---- */
.block > .container > span,
label.float,
label.checkbox-container > span {
  background: transparent !important;
  color: var(--vcl-dim) !important;
  font-weight: 600 !important;
  font-size: 0.85em !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* ---- card titles (plain, consistent with step chips) ---- */
.vcl-card-title { font-weight: 700; font-size: 1.05em; color: var(--vcl-text); margin-bottom: 6px; }

/* ---- cards ---- */
.gr-group {
  border: 1px solid var(--vcl-border) !important;
  border-radius: 14px !important;
  background: var(--vcl-panel) !important;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35) !important;
  padding: 14px 16px !important;
}

/* ---- step chips ---- */
.vcl-step { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05em;
            color: var(--vcl-text); margin-bottom: 6px; }
.vcl-step-num { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
                border-radius: 9px; font-size: 0.85em; color: #fff;
                background: linear-gradient(135deg, var(--vcl-accent), var(--vcl-accent-2));
                box-shadow: 0 0 12px var(--vcl-glow); }

/* ---- buttons ---- */
button.primary {
  background: linear-gradient(135deg, var(--vcl-accent), #8b5cf6) !important;
  border: none !important;
  box-shadow: 0 4px 18px var(--vcl-glow) !important;
}
button.primary:hover { filter: brightness(1.12); }
button.secondary { border: 1px solid var(--vcl-border) !important; }

/* ---- inputs ---- */
textarea, input[type="text"], input[type="number"], .gr-textbox textarea {
  background: var(--input-background-fill) !important;
}
textarea:focus, input:focus {
  border-color: var(--vcl-accent) !important;
  box-shadow: 0 0 0 3px rgba(124, 108, 246, 0.18) !important;
  outline: none !important;
}
textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; }

/* ---- dataframe (transcript table) ---- */
.gr-dataframe table, .gr-dataframe thead, .gr-dataframe td {
  background: var(--vcl-panel) !important;
  color: var(--vcl-text) !important;
  border-color: var(--vcl-border) !important;
}
th.header-cell {
  background: var(--vcl-panel-2) !important;
  color: var(--vcl-text) !important;
  font-weight: 650 !important;
  border-color: var(--vcl-border) !important;
}
.gr-dataframe tr:hover td { background: var(--vcl-panel-2) !important; }

/* ---- misc ---- */
footer { display: none !important; }
.vcl-footer { border-top: 1px solid var(--vcl-border); margin-top: 20px; padding-top: 10px;
              font-size: 0.82em; color: var(--vcl-dim); }
.vcl-note { font-size: 0.82em; color: var(--vcl-dim); margin-top: 6px; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2a3444; border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }
"""


def build_app(cfg: Config) -> gr.Blocks:
    """Assemble the 4-tab UI; does not launch. Wiring only — handlers live above."""
    voices = list_voices(cfg)
    default_speaker = cfg.project.speaker or (voices[0] if voices else None)

    with gr.Blocks(title="Voice Clone Lab") as app:
        gr.HTML(_HEADER_HTML)
        with gr.Row():
            speaker_dd = gr.Dropdown(
                label="Speaker",
                choices=voices,
                value=default_speaker,
                allow_custom_value=True,
                scale=3,
            )
            refresh_voices_btn = gr.Button("Refresh voices", variant="secondary", scale=1)

        with gr.Tab("1 · Setup"):
            with gr.Row(equal_height=True):
                with gr.Group():
                    gr.HTML('<div class="vcl-card-title">System check</div>')
                    check_btn = gr.Button("Run system check", variant="secondary")
                    check_out = gr.Textbox(label="System report", lines=6, max_lines=16, interactive=False)
                with gr.Group():
                    gr.HTML('<div class="vcl-card-title">Vendor setup</div>')
                    download_cb = gr.Checkbox(
                        label="Also download the base model (resolve_init_model --allow-download)", value=False
                    )
                    setup_btn = gr.Button("Run vcl setup", variant="primary")
                    setup_out = gr.Textbox(label="Setup log", lines=5, max_lines=12, interactive=False)

        with gr.Tab("2 · Data"):
            with gr.Row(equal_height=True):
                with gr.Group():
                    gr.HTML(_step_html(1, "Prepare audio"))
                    audio_in = gr.Audio(
                        label="Recording (upload or record)", type="filepath", sources=["upload", "microphone"]
                    )
                    url_in = gr.Textbox(
                        label="…or paste a URL (YouTube etc.)",
                        lines=1,
                        placeholder="https://www.youtube.com/watch?v=…",
                    )
                    pipeline_btn = gr.Button("Run pipeline (extract → clean → split)", variant="primary")
                    pipeline_log = gr.Textbox(label="Pipeline log", lines=5, max_lines=12, interactive=False)
                with gr.Group():
                    gr.HTML(_step_html(2, "Transcribe"))
                    transcribe_btn = gr.Button("Transcribe (ASR)", variant="secondary")
                    transcribe_log = gr.Textbox(label="ASR log", lines=3, max_lines=8, interactive=False)
            with gr.Group():
                gr.HTML(_step_html(3, "Review & fix transcripts"))
                transcripts_df = gr.Dataframe(
                    headers=["audio", "text"],
                    value=[],
                    type="array",
                    interactive=True,
                    wrap=True,
                    label="Transcripts — edit the text column, then Save transcripts",
                )
                with gr.Row():
                    load_tr_btn = gr.Button("Load transcripts", variant="secondary", scale=1)
                    save_tr_btn = gr.Button("Save transcripts", variant="primary", scale=1)
                transcripts_status = gr.Textbox(label="Transcript status", lines=2, max_lines=4, interactive=False)
            with gr.Group():
                gr.HTML(_step_html(4, "Build dataset"))
                dataset_btn = gr.Button("Build dataset", variant="secondary")
                dataset_out = gr.Textbox(label="Dataset summary", lines=4, max_lines=10, interactive=False)

        with gr.Tab("3 · Train"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Group():
                        gr.HTML('<div class="vcl-card-title">Run configuration</div>')
                        batch_in = gr.Number(
                            label="batch_size", value=cfg.training.batch_size, precision=0, minimum=1
                        )
                        lr_in = gr.Number(label="learning rate", value=cfg.training.lr, minimum=0)
                        epochs_in = gr.Number(label="epochs", value=cfg.training.epochs, precision=0, minimum=1)
                        attn_dd = gr.Dropdown(label="attn_implementation", choices=ATTN_CHOICES, value="auto")
                        start_btn = gr.Button("Start training", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                        overwrite_cb = gr.Checkbox(
                            label="Overwrite previous checkpoints (retraining replaces them)",
                            value=False,
                        )
                        gr.HTML(_CANCEL_NOTE_HTML)
                with gr.Column(scale=2):
                    train_status_box = gr.Textbox(label="Status", lines=2, max_lines=4, interactive=False)
                    train_log_box = gr.Textbox(label="train.log tail", lines=14, max_lines=28, interactive=False)
            train_timer = gr.Timer(2)

        with gr.Tab("4 · Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    mode_radio = gr.Radio(["fine-tuned", "zero-shot"], value="fine-tuned", label="Mode")
                    with gr.Group(visible=True) as finetuned_group:
                        with gr.Row():
                            checkpoint_dd = gr.Dropdown(
                                label="Checkpoint",
                                choices=list_checkpoints(cfg, default_speaker),
                                allow_custom_value=True,
                                scale=3,
                            )
                            refresh_ckpt_btn = gr.Button("Refresh checkpoints", variant="secondary", scale=1)
                        instruct_in = gr.Textbox(
                            label="Instruct (optional)", lines=1, placeholder="e.g. speak slowly and warmly"
                        )
                    with gr.Group(visible=False) as zeroshot_group:
                        ref_audio_in = gr.Audio(
                            label="Reference clip", type="filepath", sources=["upload", "microphone"]
                        )
                        ref_text_in = gr.Textbox(
                            label="Reference transcript (optional — improves zero-shot quality)",
                            lines=2,
                            placeholder="Transcript of the reference clip",
                        )
                    text_in = gr.Textbox(
                        label="Text", lines=4, autofocus=True, placeholder="Type the text to synthesize…"
                    )
                    language_in = gr.Textbox(label="Language", value=cfg.generation.language, lines=1)
                    generate_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.HTML('<div class="vcl-card-title">Output</div>')
                        generate_out = gr.Audio(label="Output", type="filepath", interactive=False)
                    with gr.Accordion("Advanced sampling", open=False):
                        temperature_in = gr.Slider(
                            0, 1.5, value=cfg.generation.temperature, step=0.05, label="temperature"
                        )
                        top_p_in = gr.Slider(0, 1.0, value=cfg.generation.top_p, step=0.05, label="top_p")
                        top_k_in = gr.Slider(0, 200, value=cfg.generation.top_k, step=1, label="top_k")

        gr.HTML(_FOOTER_HTML)

        # Wiring: partials bind cfg; the remaining positional args come from inputs.
        refresh_voices_btn.click(fn=partial(refresh_voices, cfg), inputs=speaker_dd, outputs=speaker_dd)

        check_btn.click(fn=partial(handle_system_check, cfg), outputs=check_out)
        setup_btn.click(fn=partial(handle_setup, cfg), inputs=download_cb, outputs=setup_out)

        pipeline_btn.click(
            fn=partial(handle_run_pipeline, cfg), inputs=[speaker_dd, audio_in, url_in], outputs=pipeline_log
        )
        transcribe_btn.click(fn=partial(handle_transcribe, cfg), inputs=speaker_dd, outputs=transcribe_log).then(
            fn=partial(load_transcripts, cfg), inputs=speaker_dd, outputs=transcripts_df
        )
        load_tr_btn.click(fn=partial(load_transcripts, cfg), inputs=speaker_dd, outputs=transcripts_df)
        save_tr_btn.click(
            fn=partial(save_transcripts, cfg), inputs=[speaker_dd, transcripts_df], outputs=transcripts_status
        )
        dataset_btn.click(fn=partial(handle_build_dataset, cfg), inputs=speaker_dd, outputs=dataset_out)

        start_btn.click(
            fn=partial(handle_train_start, cfg),
            inputs=[speaker_dd, batch_in, lr_in, epochs_in, attn_dd, overwrite_cb],
            outputs=[train_status_box, start_btn],
        )
        cancel_btn.click(fn=handle_train_cancel, outputs=train_status_box)
        train_timer.tick(
            fn=partial(poll_train, cfg),
            inputs=speaker_dd,
            outputs=[train_status_box, train_log_box, start_btn, checkpoint_dd],
        )

        speaker_dd.change(fn=partial(refresh_checkpoints, cfg), inputs=speaker_dd, outputs=checkpoint_dd)
        speaker_dd.change(fn=partial(refresh_transcript_table, cfg), inputs=speaker_dd, outputs=transcripts_df)
        refresh_ckpt_btn.click(fn=partial(refresh_checkpoints, cfg), inputs=speaker_dd, outputs=checkpoint_dd)
        mode_radio.change(fn=_mode_groups, inputs=mode_radio, outputs=[finetuned_group, zeroshot_group])
        generate_btn.click(
            fn=partial(handle_generate, cfg),
            inputs=[
                speaker_dd,
                mode_radio,
                checkpoint_dd,
                text_in,
                instruct_in,
                ref_audio_in,
                ref_text_in,
                temperature_in,
                top_p_in,
                top_k_in,
                language_in,
            ],
            outputs=generate_out,
        )

    return app
