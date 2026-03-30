"""
步驟二（前半）：OpenAI Whisper 語音轉錄
- 本地模式（預設）：使用本機 Whisper 模型，完全離線
- 遠端模式：使用 OpenAI Whisper API，速度快 10 倍以上

切換方式（在啟動 uvicorn 前設定環境變數）：
  # 本地跑（預設，不需設定任何環境變數）
  uvicorn src.platform.api:app --reload

  # 遠端 Whisper API（需要 OpenAI API Key）
  set WHISPER_MODE=remote
  set OPENAI_API_KEY=sk-...
  uvicorn src.platform.api:app --reload
"""

import os
import whisper
from config import WHISPER_MODEL


def transcribe(audio_path: str) -> list[dict]:
    """
    語音轉錄入口，依 WHISPER_MODE 環境變數自動選擇本地或遠端模式
    回傳格式：[{"start": float, "end": float, "text": str}, ...]
    """
    mode = os.getenv("WHISPER_MODE", "local").lower()
    if mode == "remote":
        print("[Transcriber] 使用遠端 OpenAI Whisper API")
        return _transcribe_remote(audio_path)
    else:
        print("[Transcriber] 使用本地 Whisper 模型")
        return _transcribe_local(audio_path)


def _transcribe_local(audio_path: str) -> list[dict]:
    """本地 Whisper 模型轉錄（離線，不需 API Key）"""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, word_timestamps=False)
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
        if seg["text"].strip()
    ]
    print(f"[Transcriber] 本地模式：共轉錄 {len(segments)} 個片段")
    return segments


def _transcribe_remote(audio_path: str) -> list[dict]:
    """
    OpenAI Whisper API 轉錄（需設定 OPENAI_API_KEY 環境變數）
    速度比本地快約 10 倍，費用約 $0.006/分鐘
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[Transcriber] 找不到 openai 套件，降級為本地模式（pip install openai）")
        return _transcribe_local(audio_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[Transcriber] 未設定 OPENAI_API_KEY，降級為本地模式")
        return _transcribe_local(audio_path)

    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        segments = [
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
            for seg in result.segments
            if seg.text.strip()
        ]
        print(f"[Transcriber] 遠端模式：共轉錄 {len(segments)} 個片段")
        return segments
    except Exception as e:
        print(f"[Transcriber] 遠端 API 失敗（{e}），降級為本地模式")
        return _transcribe_local(audio_path)
