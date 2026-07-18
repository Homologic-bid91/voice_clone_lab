"""Download voice sources from URLs (YouTube etc.) via yt-dlp.

Only download content you have the rights to use — the project's consent rule
(your own voice, or explicit written permission) applies to URLs exactly as it
does to local files. Downloaded audio is cached under the speaker's ``raw/``
dir so re-runs don't hit the network again.
"""

from __future__ import annotations

import re
from pathlib import Path

_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def is_url(value: str | None) -> bool:
    """True for http(s) URLs (anything yt-dlp might handle, YouTube included)."""
    return bool(value) and bool(re.match(r"^https?://\S+$", value.strip()))


def _ydl():
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit(
            "URL sources require yt-dlp. Install with: pip install 'voice-clone-lab[yt]'"
        ) from exc
    return yt_dlp


def download_source(url: str, raw_dir: Path, force: bool = False) -> Path:
    """Download the best audio track of ``url`` into ``raw_dir``; returns the file.

    Files are named ``youtube_<id>.<ext>`` and reused on later runs unless
    ``force=True``. Not limited to YouTube — any site yt-dlp supports works.
    """
    url = url.strip()
    if not is_url(url):
        raise SystemExit(f"Not an http(s) URL: {url!r}")
    yt_dlp = _ydl()
    raw_dir.mkdir(parents=True, exist_ok=True)

    probe_opts = {"quiet": True, "noplaylist": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise SystemExit(f"Could not fetch {url}: {exc}") from exc

    video_id = _ID_SAFE.sub("", str(info.get("id") or "")) or "download"
    title = info.get("title") or video_id
    duration = info.get("duration")
    existing = sorted(raw_dir.glob(f"youtube_{video_id}.*"))
    if existing and not force:
        print(f"Reusing cached download: {existing[0]}")
        return existing[0]

    print(f"Downloading: {title}" + (f" ({int(duration // 60)}:{int(duration % 60):02d})" if duration else ""))
    print("Reminder: only use content you have the rights to use for voice cloning.")
    outtmpl = str(raw_dir / f"youtube_{video_id}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise SystemExit(f"Download failed for {url}: {exc}") from exc

    matches = sorted(raw_dir.glob(f"youtube_{video_id}.*"))
    if not matches:
        raise SystemExit(f"Download reported success but no file appeared in {raw_dir}")
    print(f"Wrote: {matches[0]}")
    return matches[0]
