"""Tests for voice_clone_lab.ui — handlers only; no browser, no GPU, no training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from voice_clone_lab import config as config_module  # noqa: E402
from voice_clone_lab import ui  # noqa: E402
from voice_clone_lab.config import Config  # noqa: E402


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Redirect config.ROOT so paths_for/list_voices stay inside tmp_path."""
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    return tmp_path


def write_transcripts(paths, rows) -> None:
    paths.transcripts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths.transcripts_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


# --- build_app -----------------------------------------------------------------


def test_build_app_returns_blocks_without_launching():
    app = ui.build_app(Config.load())
    assert isinstance(app, gr.Blocks)


# --- speaker / checkpoint listing ----------------------------------------------


def test_list_voices(fake_root):
    voices = fake_root / "data" / "voices"
    (voices / "bob").mkdir(parents=True)
    (voices / "alice").mkdir(parents=True)
    (voices / "notavoice.txt").write_text("x", encoding="utf-8")
    cfg = Config()  # project.speaker == "default"
    assert ui.list_voices(cfg) == ["default", "alice", "bob"]


def test_list_voices_empty_when_nothing_exists(fake_root):
    cfg = Config()
    cfg.project.speaker = ""
    assert ui.list_voices(cfg) == []


def test_list_checkpoints_sorted(fake_root):
    paths = Config().paths_for("uitest")
    for name in ("checkpoint-epoch-10", "checkpoint-epoch-2", "other-dir"):
        (paths.checkpoints_dir / name).mkdir(parents=True)
    names = [Path(p).name for p in ui.list_checkpoints(Config(), "uitest")]
    assert names == ["checkpoint-epoch-2", "checkpoint-epoch-10"]


def test_list_checkpoints_empty_inputs(fake_root):
    assert ui.list_checkpoints(Config(), "  ") == []
    assert ui.list_checkpoints(Config(), "uitest") == []


# --- transcripts: load / edit / save round-trip ---------------------------------


ROWS = [
    {"audio": "data/voices/uitest/chunks/chunk_0001.wav", "text": "hello world"},
    {"audio": "data/voices/uitest/chunks/chunk_0002.wav", "text": "second clip"},
]


def test_load_transcripts_missing_raises(fake_root):
    with pytest.raises(gr.Error):
        ui.load_transcripts(Config(), "uitest")


def test_transcripts_edit_round_trip(fake_root):
    cfg = Config()
    paths = cfg.paths_for("uitest")
    write_transcripts(paths, ROWS)

    table = ui.load_transcripts(cfg, "uitest")
    assert table == [[row["audio"], row["text"]] for row in ROWS]

    table[0][1] = "edited text"
    status = ui.save_transcripts(cfg, "uitest", table)

    # The review TSV is written in the exact `audio\ttext` format...
    tsv_lines = paths.review_tsv.read_text(encoding="utf-8").splitlines()
    assert tsv_lines[0] == "audio\ttext"
    assert tsv_lines[1] == f"{ROWS[0]['audio']}\tedited text"
    # ...and apply_review wrote the edit back to transcripts.jsonl.
    applied = [
        json.loads(line)
        for line in paths.transcripts_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert applied == [
        {"audio": ROWS[0]["audio"], "text": "edited text"},
        ROWS[1],
    ]
    assert "2" in status


def test_save_transcripts_skips_empty_cells_via_apply_review(fake_root):
    cfg = Config()
    paths = cfg.paths_for("uitest")
    write_transcripts(paths, ROWS)
    table = ui.load_transcripts(cfg, "uitest")
    table[1][1] = "   "  # empty text cell -> apply_review skips the row
    status = ui.save_transcripts(cfg, "uitest", table)
    applied = [
        json.loads(line)
        for line in paths.transcripts_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert applied == [ROWS[0]]
    assert "1" in status


def test_save_transcripts_empty_table_raises(fake_root):
    with pytest.raises(gr.Error):
        ui.save_transcripts(Config(), "uitest", [])


def test_load_transcripts_silent(fake_root):
    # No speaker / bad speaker / voice without transcripts -> empty, never raises.
    assert ui.load_transcripts_silent(Config(), None) == []
    assert ui.load_transcripts_silent(Config(), "a/b") == []
    assert ui.load_transcripts_silent(Config(), "ghost") == []
    # Voice with transcripts -> rows.
    paths = Config().paths_for("uitest")
    write_transcripts(paths, ROWS)
    assert ui.load_transcripts_silent(Config(), "uitest") == [[row["audio"], row["text"]] for row in ROWS]


def test_save_transcripts_rejects_other_voice_rows(fake_root):
    """Stale rows from a previously selected voice must not land in this speaker's file."""
    cfg = Config()
    paths = cfg.paths_for("uitest")
    write_transcripts(paths, ROWS)
    stale = [["data/voices/someoneelse/chunks/chunk_0001.wav", "stale row"]]
    with pytest.raises(gr.Error, match="different voice"):
        ui.save_transcripts(cfg, "uitest", stale)
    # The current speaker's transcripts are untouched.
    applied = [json.loads(line) for line in paths.transcripts_jsonl.read_text(encoding="utf-8").splitlines()]
    assert applied == ROWS


# --- guards ---------------------------------------------------------------------


def test_transcribe_without_chunks_raises(fake_root):
    with pytest.raises(gr.Error, match="pipeline"):
        ui.handle_transcribe(Config(), "uitest")


def test_pipeline_requires_source(fake_root):
    with pytest.raises(gr.Error):
        ui.handle_run_pipeline(Config(), "uitest", None, None)


def test_pipeline_rejects_garbage_url(fake_root):
    with pytest.raises(gr.Error):
        ui.handle_run_pipeline(Config(), "uitest", None, "not a url")


def test_speaker_required(fake_root):
    with pytest.raises(gr.Error):
        ui.load_transcripts(Config(), "  ")
    with pytest.raises(gr.Error):
        ui.load_transcripts(Config(), "a/b")


def test_build_dataset_without_transcripts_raises(fake_root):
    with pytest.raises(gr.Error):
        ui.handle_build_dataset(Config(), "uitest")


# --- train state -----------------------------------------------------------------


def test_train_cancel_without_run():
    assert "No training run" in ui.handle_train_cancel()


def test_train_log_tail(fake_root):
    cfg = Config()
    assert ui.train_log_tail(cfg, "uitest") == ""  # no log yet
    paths = cfg.paths_for("uitest")
    paths.checkpoints_dir.mkdir(parents=True)
    (paths.checkpoints_dir / "train.log").write_text("line1\nline2\n", encoding="utf-8")
    assert ui.train_log_tail(cfg, "uitest") == "line1\nline2\n"
    assert ui.train_log_tail(cfg, "uitest", max_chars=6) == "line2\n"


class _FakeThread:
    """Records start() without running anything (no real training in tests)."""

    started: bool = False

    def __init__(self, target=None, args=(), daemon=None):
        self._target = target

    def start(self):
        type(self).started = True

    def is_alive(self):
        return False


def test_train_start_refuses_retrain_without_overwrite(fake_root, monkeypatch):
    paths = Config().paths_for("uitest")
    (paths.checkpoints_dir / "checkpoint-epoch-0").mkdir(parents=True)
    _FakeThread.started = False
    monkeypatch.setattr(ui.threading, "Thread", _FakeThread)
    warned = {}
    monkeypatch.setattr(ui.gr, "Warning", lambda msg, **kw: warned.setdefault("msg", msg))
    msg, update = ui.handle_train_start(Config(), "uitest", 8, 2e-6, 1, "auto", False)
    assert "Overwrite previous checkpoints" in warned["msg"]
    assert "Not started" in msg
    assert update["interactive"] is True  # Start stays clickable
    assert not _FakeThread.started  # nothing was launched


def test_train_start_launches_with_overwrite(fake_root, monkeypatch):
    paths = Config().paths_for("uitest")
    (paths.checkpoints_dir / "checkpoint-epoch-0").mkdir(parents=True)
    _FakeThread.started = False
    monkeypatch.setattr(ui.threading, "Thread", _FakeThread)
    msg, update = ui.handle_train_start(Config(), "uitest", 8, 2e-6, 1, "auto", True)
    assert "Training started" in msg
    assert _FakeThread.started
    assert update["interactive"] is False


def test_train_start_launches_when_no_checkpoints(fake_root, monkeypatch):
    _FakeThread.started = False
    monkeypatch.setattr(ui.threading, "Thread", _FakeThread)
    msg, _ = ui.handle_train_start(Config(), "uitest", 8, 2e-6, 1, "auto", False)
    assert "Training started" in msg
    assert _FakeThread.started


def test_poll_train_returns_four_outputs(fake_root):
    status, log, start_update, ckpt_update = ui.poll_train(Config(), "uitest")
    assert isinstance(status, str)
    assert isinstance(log, str)
    assert "interactive" in start_update
