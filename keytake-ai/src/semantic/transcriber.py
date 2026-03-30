"""
步驟二（前半）：OpenAI Whisper 語音轉錄
- 輸出帶有精確時間戳記的逐字稿
對應計畫書 4.2 步驟二
"""

import whisper
from config import WHISPER_MODEL


def transcribe(audio_path: str) -> list[dict]:
    """
    使用 Whisper 轉錄音訊，回傳帶時間戳記的片段列表
    每個元素格式：{"start": float, "end": float, "text": str}
    """
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, word_timestamps=False)

    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
        if seg["text"].strip()  # 過濾 Whisper 幻覺產生的空字串片段
    ]
    print(f"[Transcriber] 共轉錄 {len(segments)} 個片段")
    return segments
