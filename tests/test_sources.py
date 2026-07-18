"""Tests for sources.py — URL detection and yt-dlp download caching (no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_clone_lab.sources import download_source, is_url


class TestIsUrl:
    @pytest.mark.parametrize(
        "value",
        [
            "https://www.youtube.com/watch?v=abc123",
            "http://example.com/audio.mp3",
            "  https://youtu.be/abc123  ",
        ],
    )
    def test_urls(self, value):
        assert is_url(value)

    @pytest.mark.parametrize("value", [None, "", "data/raw/x.wav", "/tmp/x.mp4", "ftp://x/y", "not a url"])
    def test_non_urls(self, value):
        assert not is_url(value)


def _fake_ydl(monkeypatch, info=None, writes=("m4a",)):
    """Install a fake yt_dlp module whose YoutubeDL returns canned info and 'downloads' files."""
    info = info or {"id": "vid123", "title": "Test Video", "duration": 61}
    downloads = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            if download:
                downloads.append(url)
                outtmpl = self.opts["outtmpl"]
                for ext in writes:
                    Path(outtmpl.replace("%(ext)s", ext)).write_bytes(b"fake")
            return info

    fake_module = MagicMock()
    fake_module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    return downloads


class TestDownloadSource:
    def test_downloads_and_caches(self, tmp_path, monkeypatch):
        downloads = _fake_ydl(monkeypatch)
        first = download_source("https://www.youtube.com/watch?v=vid123", tmp_path)
        assert first == tmp_path / "youtube_vid123.m4a"
        assert len(downloads) == 1
        # Second call reuses the cache — no new download.
        second = download_source("https://www.youtube.com/watch?v=vid123", tmp_path)
        assert second == first
        assert len(downloads) == 1

    def test_force_redownloads(self, tmp_path, monkeypatch):
        downloads = _fake_ydl(monkeypatch)
        download_source("https://www.youtube.com/watch?v=vid123", tmp_path)
        download_source("https://www.youtube.com/watch?v=vid123", tmp_path, force=True)
        assert len(downloads) == 2

    def test_rejects_non_url(self, tmp_path):
        with pytest.raises(SystemExit, match="Not an http"):
            download_source("data/raw/x.wav", tmp_path)

    def test_missing_yt_dlp(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "yt_dlp", None)
        with pytest.raises(SystemExit, match="yt-dlp"):
            download_source("https://www.youtube.com/watch?v=vid123", tmp_path)

    def test_fetch_failure(self, tmp_path, monkeypatch):
        class BoomYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                raise RuntimeError("HTTP 403")

        fake_module = MagicMock()
        fake_module.YoutubeDL = BoomYDL
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
        with pytest.raises(SystemExit, match="Could not fetch"):
            download_source("https://www.youtube.com/watch?v=vid123", tmp_path)
