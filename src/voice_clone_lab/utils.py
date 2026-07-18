"""Shared IO helpers and the overwrite-safety convention.

Every pipeline step that writes files refuses to overwrite existing outputs
unless ``force=True``. Artifacts that store audio paths (chunk metadata,
transcripts, dataset rows) always store project-relative paths; readers accept
both relative paths and legacy absolute paths via :func:`resolve_artifact_path`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import ROOT, project_path, rel_to_root  # re-exported for convenience

__all__ = [
    "ROOT",
    "project_path",
    "rel_to_root",
    "resolve_artifact_path",
    "ensure_parent",
    "refuse_overwrite",
    "refuse_any_existing",
    "read_jsonl",
    "write_jsonl",
    "write_json",
    "command_exists",
    "run",
]


def resolve_artifact_path(value: str | Path) -> Path:
    """Resolve a path stored in an artifact file.

    Relative paths are resolved against the project root; absolute paths
    (written by the legacy scripts/) pass through unchanged.
    """
    return project_path(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def refuse_overwrite(path: Path, force: bool, label: str) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"{label} already exists: {path}\nRefusing to overwrite. Re-run with --force."
        )


def refuse_any_existing(paths: list[Path], force: bool, label: str) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        shown = "\n".join(f"  {p}" for p in existing[:20])
        extra = f"\n  ... and {len(existing) - 20} more" if len(existing) > 20 else ""
        raise SystemExit(
            f"{label} already exist:\n{shown}{extra}\nRefusing to overwrite. Re-run with --force."
        )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict], force: bool = False) -> None:
    refuse_overwrite(path, force, "JSONL file")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj, force: bool = False) -> None:
    refuse_overwrite(path, force, "JSON file")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command quietly and return stripped stdout, or None on failure."""
    try:
        return subprocess.check_output(
            [str(c) for c in cmd], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
