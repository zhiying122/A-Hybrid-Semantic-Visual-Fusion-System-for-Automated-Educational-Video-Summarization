"""Whisper 語音轉錄：支援本地模型與 OpenAI API。"""

import os
from pathlib import Path
from typing import Any

from config import WHISPER_MODEL

try:
    import whisper as _whisper
    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    _whisper = None
    WHISPER_LOCAL_AVAILABLE = False


_LOCAL_MODELS: dict[str, Any] = {}


def _deduplicate_segments(
    segments: list[dict], max_gap_sec: float = 2.0
) -> list[dict]:
    """合併時間相近且文字相同的連續片段，避免把遠距重複內容拉成長片段。"""
    if not segments:
        return []

    cleaned = []
    for segment in segments:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
            text = " ".join(str(segment.get("text", "")).split())
        except (KeyError, TypeError, ValueError):
            continue

        if not text or start < 0 or end <= start:
            continue

        cleaned.append({"start": start, "end": end, "text": text})

    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    if not cleaned:
        return []

    deduped = [cleaned[0]]
    removed = 0

    for segment in cleaned[1:]:
        previous = deduped[-1]
        same_text = segment["text"].casefold() == previous["text"].casefold()
        gap = segment["start"] - previous["end"]

        if same_text and gap <= max_gap_sec:
            previous["end"] = max(previous["end"], segment["end"])
            removed += 1
        else:
            deduped.append(segment)

    if removed:
        print(f"[Transcriber] 已合併 {removed} 個時間相近的重複片段")
    return deduped


def transcribe(audio_path: str) -> list[dict]:
    """轉錄音訊，回傳 start、end、text 欄位組成的片段列表。"""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音訊檔案：{path}")

    mode = os.getenv("WHISPER_MODE", "local").strip().lower()
    if mode not in {"local", "remote"}:
        raise ValueError("WHISPER_MODE 必須是 local 或 remote")

    if mode == "remote":
        print("[Transcriber] 使用 OpenAI Whisper API")
        segments = _transcribe_remote(str(path))
    else:
        print("[Transcriber] 使用本地 Whisper 模型")
        segments = _transcribe_local(str(path))

    return _deduplicate_segments(segments)


def _get_local_model():
    """同一個程序內重用已載入的 Whisper 模型，避免重複分配記憶體。"""
    if not WHISPER_LOCAL_AVAILABLE:
        raise ImportError("openai-whisper 未安裝，請執行：pip install openai-whisper")
    model_name = str(WHISPER_MODEL)
    if model_name not in _LOCAL_MODELS:
        _LOCAL_MODELS[model_name] = _whisper.load_model(model_name)
    return _LOCAL_MODELS[model_name]


def _transcribe_local(audio_path: str) -> list[dict]:
    """使用本地 Whisper 模型轉錄。若 whisper 未安裝則拋 ImportError。"""
    model = _get_local_model()
    result = model.transcribe(audio_path, word_timestamps=False)

    segments = []
    for item in result.get("segments", []):
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text})

    print(f"[Transcriber] 本地模式：取得 {len(segments)} 個片段")
    return segments


def _transcribe_remote(audio_path: str) -> list[dict]:
    """使用 OpenAI Whisper API；遠端失敗時嘗試本地模型。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[Transcriber] 未安裝 openai 套件，改用本地模式")
        return _transcribe_local(audio_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[Transcriber] 未設定 OPENAI_API_KEY，改用本地模式")
        return _transcribe_local(audio_path)

    try:
        client = OpenAI(
            api_key=api_key,
            timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "60")),
            max_retries=2,
        )
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=os.getenv("WHISPER_API_MODEL", "whisper-1"),
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for item in result.segments or []:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            try:
                start = float(item.start)
                end = float(item.end)
            except (AttributeError, TypeError, ValueError):
                continue
            if start < 0 or end <= start:
                continue
            segments.append({"start": start, "end": end, "text": text})

        print(f"[Transcriber] 遠端模式：取得 {len(segments)} 個片段")
        return segments

    except Exception as exc:
        print(f"[Transcriber] 遠端 API 失敗：{exc}；嘗試改用本地模式")
        return _transcribe_local(audio_path)
