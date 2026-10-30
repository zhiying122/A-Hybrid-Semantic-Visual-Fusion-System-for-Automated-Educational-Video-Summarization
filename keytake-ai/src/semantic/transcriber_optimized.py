"""Whisper 語音轉錄：支援本地模型與 OpenAI API。"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

from config import WHISPER_MODEL

try:
    import whisper as _whisper
    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    _whisper = None
    WHISPER_LOCAL_AVAILABLE = False


_LOCAL_MODELS: dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """Merge same-text segments only when their time ranges almost fully overlap."""
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
        overlap = min(segment["end"], previous["end"]) - max(
            segment["start"], previous["start"]
        )
        shorter_duration = min(
            segment["end"] - segment["start"],
            previous["end"] - previous["start"],
        )
        # 只有文字相同且較短片段至少 80% 被重疊覆蓋才合併，避免誤刪相鄰重複句。
        duplicate_overlap = (
            overlap > 0
            and shorter_duration > 0
            and overlap / shorter_duration >= 0.8
        )

        if same_text and duplicate_overlap:
            previous["end"] = max(previous["end"], segment["end"])
            removed += 1
        else:
            deduped.append(segment)

    if removed:
        logger.info("[Transcriber] 已合併 %d 個高度重疊的重複片段", removed)
    return deduped


def transcribe(audio_path: str) -> list[dict]:
    """轉錄音訊，回傳 start、end、text 欄位組成的片段列表。"""
    return transcribe_with_metadata(audio_path)["segments"]


def transcribe_with_metadata(audio_path: str) -> dict:
    """Return normalized transcript segments plus the backend and fallback reason."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音訊檔案：{path}")

    mode = os.getenv("WHISPER_MODE", "local").strip().lower()
    if mode not in {"local", "remote"}:
        raise ValueError("WHISPER_MODE 必須是 local 或 remote")

    metadata = {
        "requested_backend": mode,
        "backend": None,
        "model": None,
        "fallback_reason": None,
    }
    # remote 會把音訊送到 OpenAI；local 模式不做遠端回退，缺模型時直接報錯。
    if mode == "remote":
        logger.info(
            "[Transcriber] 使用 OpenAI Whisper API model=%s",
            os.getenv("WHISPER_API_MODEL", "whisper-1"),
        )
        segments = _transcribe_remote(str(path), metadata)
    else:
        logger.info("[Transcriber] 使用本地 Whisper model=%s", WHISPER_MODEL)
        segments = _transcribe_local(str(path), metadata)

    return {
        "segments": _deduplicate_segments(segments),
        "metadata": metadata,
    }


def _get_local_model():
    """同一個程序內重用已載入的 Whisper 模型，避免重複分配記憶體。"""
    if not WHISPER_LOCAL_AVAILABLE:
        raise ImportError("openai-whisper 未安裝，請執行：pip install openai-whisper")
    model_name = str(WHISPER_MODEL)
    if model_name not in _LOCAL_MODELS:
        _LOCAL_MODELS[model_name] = _whisper.load_model(model_name)
    return _LOCAL_MODELS[model_name]


def _transcribe_local(
    audio_path: str,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """使用本地 Whisper 模型轉錄。若 whisper 未安裝則拋 ImportError。"""
    if metadata is not None:
        metadata["backend"] = "local"
        metadata["model"] = str(WHISPER_MODEL)
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

    logger.info("[Transcriber] 本地模式取得 %d 個片段", len(segments))
    return segments


def _transcribe_remote(
    audio_path: str,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """使用 OpenAI Whisper API；遠端失敗時嘗試本地模型。"""
    try:
        from openai import OpenAI
    except ImportError:
        reason = "openai 套件未安裝"
        if metadata is not None:
            metadata["fallback_reason"] = reason
        logger.warning("[Transcriber] %s，改用本地模式", reason)
        return _transcribe_local(audio_path, metadata)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        reason = "未設定 OPENAI_API_KEY"
        if metadata is not None:
            metadata["fallback_reason"] = reason
        logger.warning("[Transcriber] %s，改用本地模式", reason)
        return _transcribe_local(audio_path, metadata)

    try:
        model_name = os.getenv("WHISPER_API_MODEL", "whisper-1")
        client = OpenAI(
            api_key=api_key,
            timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "60")),
            max_retries=2,
        )
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=model_name,
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

        logger.info("[Transcriber] 遠端模式取得 %d 個片段", len(segments))
        if metadata is not None:
            metadata["backend"] = "remote"
            metadata["model"] = model_name
        return segments

    except Exception as exc:
        if metadata is not None:
            metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[Transcriber] 遠端 API 失敗 (%s): %s；嘗試改用本地模式",
            type(exc).__name__,
            exc,
        )
        return _transcribe_local(audio_path, metadata)
