#!/usr/bin/env python3
"""Web app for video speech-to-text extraction."""

import os
import sys
import shutil
import tempfile
from pathlib import Path

from flask import Flask, render_template, request

import downloader
import formatter
import transcriber

# 国内镜像，避免 HuggingFace 被墙
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# 启动时预加载模型
print("正在加载 Whisper 模型...", file=sys.stderr)
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            error = "请输入视频URL"
        else:
            try:
                platform = downloader.validate_url(url)
            except downloader.UnsupportedURLError as e:
                error = str(e)
            else:
                # 下载音频
                tmp_dir = Path(tempfile.mkdtemp(prefix="video_text_"))
                wav_path = None
                try:
                    wav_path = downloader.download_audio(url, tmp_dir)

                    # 转写
                    segments = transcriber.transcribe(wav_path)

                    if not segments:
                        error = "未检测到语音内容"
                    else:
                        result = formatter.format_text(segments, interval=20)

                except downloader.DownloadError as e:
                    error = f"下载失败: {e}"
                except transcriber.TranscriptionError as e:
                    error = f"转写失败: {e}"
                finally:
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 将结果按换行分割，方便前端渲染
    result_lines = result.split("\n") if result else []
    return render_template("index.html", url=url, result_lines=result_lines, error=error)


if __name__ == "__main__":
    get_model()
    print("模型加载完成，服务启动: http://127.0.0.1:5000", file=sys.stderr)
    app.run(host="0.0.0.0", port=5000, debug=False)
