"""Speech-to-text transcription using faster-whisper."""

from pathlib import Path


class TranscriptionError(Exception):
    """Transcription failed (model load error, corrupt audio, etc.)."""


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[dict]:
    """
    Transcribe an audio file (WAV, 16kHz mono) to text segments with timestamps.
    Returns list of dicts: [{"start": 0.0, "end": 2.5, "text": "你好"}, ...]
    """
    from faster_whisper import WhisperModel

    print(f"  加载模型 '{model_size}' ({device}/{compute_type})...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        raise TranscriptionError(f"模型加载失败: {e}")

    print("  语音识别中...")
    try:
        raw_segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=language,
            vad_filter=True,
            without_timestamps=False,
        )
    except Exception as e:
        raise TranscriptionError(f"转写失败: {e}")

    segments = []
    for seg in raw_segments:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    print(f"  检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"  识别出 {len(segments)} 个语音片段，总时长 {segments[-1]['end']:.0f} 秒" if segments else "  未检测到语音")
    return segments
