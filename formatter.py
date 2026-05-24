"""Format transcription segments with periodic timestamp markers."""

import json


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def inject_timestamps(segments: list[dict], interval: int = 20) -> list[dict]:
    """
    Insert timestamp-only marker entries at least every `interval` seconds.
    Returns a mixed list where each item is either:
      {"type": "marker", "time": float}
      {"type": "segment", "start": float, "end": float, "text": str}
    Markers are placed at the nearest segment boundary, never mid-phrase.
    """
    if not segments:
        return []

    total_duration = max(s["end"] for s in segments)
    result = []
    next_marker = 0

    for i, seg in enumerate(segments):
        while next_marker <= seg["start"] <= total_duration:
            result.append({"type": "marker", "time": next_marker})
            next_marker += interval

        result.append({
            "type": "segment",
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })

    return result


def format_text(segments: list[dict], interval: int = 20) -> str:
    """Format as plain text with [HH:MM:SS] markers every `interval` seconds."""
    mixed = inject_timestamps(segments, interval)
    lines = []

    # Group consecutive segments under each marker
    current_marker = None
    buffer = []

    def flush():
        if current_marker is not None and buffer:
            lines.append(f"[{format_timestamp(current_marker)}]")
            lines.append("".join(buffer))
            lines.append("")

    for item in mixed:
        if item["type"] == "marker":
            flush()
            current_marker = item["time"]
            buffer = []
        else:
            buffer.append(item["text"])

    flush()

    # If there's no marker at all (short video), add one at the start
    if not lines and mixed:
        lines.insert(0, f"[00:00:00]")
        lines.append("".join(item["text"] for item in mixed if item["type"] == "segment"))

    return "\n".join(lines).strip()


def format_srt(segments: list[dict]) -> str:
    """Format as standard SRT subtitle file."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _srt_timestamp(seg["start"])
        end_ts = _srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).strip()


def format_json_output(url: str, platform: str, segments: list[dict],
                       model: str, language: str, interval: int) -> str:
    """Format as JSON with metadata and marker positions."""
    mixed = inject_timestamps(segments, interval)
    duration = max(s["end"] for s in segments) if segments else 0

    output = {
        "url": url,
        "platform": platform,
        "duration_seconds": round(duration, 2),
        "model": model,
        "language": language,
        "timestamp_interval": interval,
        "segments": segments,
        "markers": [
            {"time": item["time"]}
            for item in mixed if item["type"] == "marker"
        ],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def _srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
