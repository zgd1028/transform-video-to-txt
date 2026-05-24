"""Speech-to-text transcription using faster-whisper."""

import os
import re
from pathlib import Path

# 国内用户默认使用镜像，避免 HuggingFace 被墙
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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

    segments = _add_punctuation(segments)

    print(f"  检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"  识别出 {len(segments)} 个语音片段，总时长 {segments[-1]['end']:.0f} 秒" if segments else "  未检测到语音")
    return segments


def _add_punctuation(segments: list[dict]) -> list[dict]:
    """根据片段间停顿时间自动添加中文标点符号。"""
    if not segments:
        return segments

    # 中文标点符号集合
    cn_punct = "，。！？；：、\"\"''（）…—～"

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        # 去掉尾部已有标点，避免重复
        text = text.rstrip(cn_punct + ",.!?;: ")

        # 计算到下一个片段的停顿时间
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - seg["end"]
        else:
            gap = 999  # 最后一句

        # 根据停顿时间选择标点
        if gap > 1.2:
            punct = "。"
        elif gap > 0.7:
            punct = "！" if _is_exclamatory(text) else "。"
        elif gap > 0.35:
            punct = "，"
        elif gap > 0.15:
            punct = "；" if len(text) > 8 else "，"
        else:
            punct = "，"

        text += punct
        segments[i]["text"] = text

    # 最后一句用句号
    last_text = segments[-1]["text"].rstrip(cn_punct + ",.!?;: ")
    segments[-1]["text"] = last_text + "。"

    return segments


def _is_exclamatory(text: str) -> bool:
    """判断句子是否有感叹语气。"""
    exclam_words = [
        "啊", "哇", "太棒", "好极了", "天哪", "我的天", "太.*了",
        "真.*啊", "多么", "多么.*啊", "竟然", "居然", "不可思议",
        "厉害", "牛逼", "牛", "绝了", "太强", "无敌",
    ]
    for w in exclam_words:
        if re.search(w, text):
            return True
    return False
