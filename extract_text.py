#!/usr/bin/env python3
"""Extract spoken text from Bilibili/Douyin videos with timestamp annotations.

Usage:
  python extract_text.py <url> [--model small] [--format text|srt|json]
      [--interval 20] [--output file.txt] [--keep-audio] [--device cpu|cuda]
      [--language zh]

Timestamps are inserted at least every 20 seconds by default.
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import downloader
import formatter
import transcriber


def main():
    parser = argparse.ArgumentParser(
        description="从B站/抖音视频中提取语音文字，带时间戳标注"
    )
    parser.add_argument("url", help="视频URL (Bilibili或抖音网页版)")
    parser.add_argument(
        "--model", "-m", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper模型大小 (默认: small)"
    )
    parser.add_argument(
        "--format", "-f", default="text", choices=["text", "srt", "json"],
        help="输出格式 (默认: text)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=20,
        help="时间戳标注间隔(秒) (默认: 20)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="输出文件路径 (默认: 输出到终端)"
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="保留下载的音频文件"
    )
    parser.add_argument(
        "--device", "-d", default="cpu", choices=["cpu", "cuda"],
        help="推理设备 (默认: cpu)"
    )
    parser.add_argument(
        "--language", "-l", default="zh",
        help="语言代码 (默认: zh)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示详细信息"
    )

    args = parser.parse_args()
    start_time = time.time()

    # Step 1: Validate URL
    try:
        platform = downloader.validate_url(args.url)
    except downloader.UnsupportedURLError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"平台: {platform}", file=sys.stderr)

    # Step 2: Download audio
    tmp_dir = Path(tempfile.mkdtemp(prefix="video_text_"))
    wav_path = None
    try:
        print("正在下载视频音频...", file=sys.stderr)
        wav_path = downloader.download_audio(args.url, tmp_dir)

        # Step 3: Transcribe
        segments = transcriber.transcribe(
            wav_path,
            model_size=args.model,
            language=args.language,
            device=args.device,
        )

        if not segments:
            print("警告: 未检测到语音内容", file=sys.stderr)
            sys.exit(0)

        # Step 4: Format output
        if args.format == "text":
            output = formatter.format_text(segments, args.interval)
        elif args.format == "srt":
            output = formatter.format_srt(segments)
        elif args.format == "json":
            output = formatter.format_json_output(
                args.url, platform, segments, args.model,
                args.language, args.interval
            )
        else:
            output = formatter.format_text(segments, args.interval)

        # Step 5: Write output
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"已保存到: {args.output}", file=sys.stderr)
        else:
            print(output)

        elapsed = time.time() - start_time
        print(f"处理完成，耗时 {elapsed:.1f} 秒", file=sys.stderr)

    except downloader.DownloadError as e:
        print(f"下载错误: {e}", file=sys.stderr)
        sys.exit(2)
    except transcriber.TranscriptionError as e:
        print(f"转写错误: {e}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\n已取消", file=sys.stderr)
        sys.exit(130)
    finally:
        # Cleanup temp files
        if args.keep_audio and wav_path and wav_path.exists():
            print(f"音频文件保留在: {wav_path}", file=sys.stderr)
        elif tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
