"""Bilibili/Douyin URL validation and audio download via yt-dlp."""

import re
import tempfile
from pathlib import Path

import yt_dlp


class UnsupportedURLError(Exception):
    """URL does not match any supported platform."""


class DownloadError(Exception):
    """Audio download or extraction failed."""


def validate_url(url: str) -> str:
    """Return platform name ('bilibili' or 'douyin') or raise UnsupportedURLError."""
    url_lower = url.lower()
    if any(d in url_lower for d in ["bilibili.com/video/", "bilibili.com/bangumi/", "b23.tv/"]):
        return "bilibili"
    if any(d in url_lower for d in ["douyin.com/video/", "douyin.com/user/"]):
        return "douyin"
    raise UnsupportedURLError(f"不支持的URL: {url}\n目前支持: Bilibili (bilibili.com) 和 抖音 (douyin.com)")


def _progress_hook(d):
    """yt-dlp progress hook — prints download status to stderr."""
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?").strip()
        speed = d.get("_speed_str", "?").strip()
        eta = d.get("_eta_str", "?").strip()
        print(f"\r  下载音频: {pct} 速度: {speed} 剩余: {eta}", end="", flush=True)
    elif d["status"] == "finished":
        print("\r  下载完成，正在处理音频...")


def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download best audio stream from the given video URL.
    Extract to 16kHz mono WAV and return path to the result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": None,
            }
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
        "retries": 3,
        "fragment_retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "unknown")
            wav_path = output_dir / f"{video_id}.wav"
            if not wav_path.exists():
                raise DownloadError(f"音频文件未生成: {wav_path}")
            print(f"\r  音频下载完成: {wav_path.name}")
            return wav_path
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "video unavailable" in msg or "private" in msg:
            raise DownloadError(f"视频不可用: {e}")
        if "ffmpeg" in msg or "ffprobe" in msg:
            raise DownloadError(f"未找到 FFmpeg。请安装: winget install ffmpeg\n原错误: {e}")
        raise DownloadError(f"下载失败: {e}")
