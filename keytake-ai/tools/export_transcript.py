"""
將 Whisper 轉錄結果匯出為標註工具可用的 JSON 格式
用法：python tools/export_transcript.py <audio_or_video_path> --output transcript.json
"""

import json
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.semantic.transcriber import transcribe
from src.preprocessing.preprocessor import extract_audio


def main():
    parser = argparse.ArgumentParser(description="匯出 Whisper 逐字稿供標註使用")
    parser.add_argument("input", help="影片或音訊路徑")
    parser.add_argument("--output", default="transcript.json", help="輸出 JSON 路徑")
    args = parser.parse_args()

    # 若輸入為影片，先提取音訊
    ext = os.path.splitext(args.input)[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        audio_path = args.input.replace(ext, "_audio.wav")
        print(f"提取音訊中...")
        extract_audio(args.input, audio_path)
    else:
        audio_path = args.input

    print(f"轉錄中（模型：{__import__('config').WHISPER_MODEL}）...")
    segments = transcribe(audio_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"完成！共 {len(segments)} 個片段，已儲存至 {args.output}")


if __name__ == "__main__":
    main()
