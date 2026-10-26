"""
語音轉錄模組

以 Whisper 將教學影片音訊轉為帶時間戳的片段列表。
支援本地模式（離線）與遠端 OpenAI API 模式，透過環境變數 WHISPER_MODE 切換。
內建重複片段合併，過濾長時間靜音造成的幻覺輸出。
"""

import os
import whisper
from config import WHISPER_MODEL


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """
    Whisper 幻覺過濾：合併連續重複的片段。
    當影片有長時間靜音時，Whisper 可能重複輸出上一句文字。
    """
    if not segments:
        return segments
    deduped = [segments[0]]
    for seg in segments[1:]:
        if seg["text"] == deduped[-1]["text"]:
            # 合併：延伸前一段的結束時間
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)
    removed = len(segments) - len(deduped)
    if removed > 0:
        print(f"[Transcriber] 幻覺過濾：合併了 {removed} 個重複片段")
    return deduped


def transcribe(audio_path: str) -> list[dict]:
    """
    語音轉錄入口，依 WHISPER_MODE 環境變數自動選擇本地或遠端模式
    回傳格式：[{"start": float, "end": float, "text": str}, ...]
    """
    mode = os.getenv("WHISPER_MODE", "local").lower()
    if mode == "remote":
        print("[Transcriber] 使用遠端 OpenAI Whisper API")
        segments = _transcribe_remote(audio_path)
    else:
        print("[Transcriber] 使用本地 Whisper 模型")
        segments = _transcribe_local(audio_path)
    return _deduplicate_segments(segments)


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
