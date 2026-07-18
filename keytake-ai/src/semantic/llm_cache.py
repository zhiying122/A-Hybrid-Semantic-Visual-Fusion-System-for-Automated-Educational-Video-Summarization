"""
LLM 評分結果快取模組。

本模組解決 LLM 評分的非確定性與 API 成本問題。透過將評分結果以文本內容的雜湊值
作為鍵進行快取，重複實驗可產生一致的結果，同時節省 API 呼叫次數與費用。

快取以 JSON 檔案作為持久化儲存，支援執行緒安全、延遲載入、批次寫入與 TTL 過期機制。
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class LLMCache:
    """LLM 評分結果快取類別。

    透過對輸入文本計算 SHA-256 雜湊值作為快取鍵，將評分結果儲存於 JSON 檔案中。
    支援延遲載入、批次寫入、TTL 過期與執行緒安全操作。
    """

    def __init__(self, cache_dir: str = "cache", max_age_days: Optional[float] = None,
                 flush_every: int = 10):
        """初始化快取實例。

        Args:
            cache_dir: 快取目錄路徑，預設為 "cache"
            max_age_days: 快取條目最大存活天數，None 表示永不過期
            flush_every: 每累積 N 次寫入後自動刷新到磁碟，預設為 10
        """
        # 快取目錄與檔案路徑
        self._cache_dir = Path(cache_dir)
        self._cache_file = self._cache_dir / "llm_scores.json"

        # TTL 設定（天數）
        self._max_age_days = max_age_days

        # 批次寫入設定
        self._flush_every = flush_every
        self._pending_writes = 0

        # 執行緒鎖，確保執行緒安全
        self._lock = threading.Lock()

        # 延遲載入旗標，首次存取時才讀取 JSON 檔案
        self._loaded = False
        self._data: dict = {}

        # 建立快取目錄（若不存在）
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        """確保快取資料已從磁碟載入（延遲載入）。"""
        if self._loaded:
            return

        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # 檔案損壞或讀取失敗，重新初始化為空快取
                self._data = {}
        else:
            self._data = {}

        self._loaded = True

    def _hash_key(self, text: str) -> str:
        """計算文本的 SHA-256 雜湊值，取前 16 個十六進位字元作為鍵。

        Args:
            text: 輸入文本

        Returns:
            前 16 字元的十六進位雜湊字串
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _is_expired(self, entry: dict) -> bool:
        """檢查快取條目是否已過期。

        Args:
            entry: 快取條目字典，需包含 timestamp 欄位

        Returns:
            若已過期回傳 True，否則回傳 False
        """
        # 未設定 TTL 則永不過期
        if self._max_age_days is None:
            return False

        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            # 確保時區資訊一致
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_days = (now - entry_time).total_seconds() / 86400
            return age_days > self._max_age_days
        except (KeyError, ValueError):
            # 缺少時間戳或格式錯誤，視為已過期
            return True

    def _flush_to_disk(self) -> None:
        """將記憶體中的快取資料寫入磁碟。"""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError:
            # 寫入失敗時靜默處理，避免中斷主流程
            pass

    def get(self, text: str) -> Optional[dict]:
        """取得快取中的評分結果。

        Args:
            text: 輸入文本

        Returns:
            快取的結果字典（包含 score, stage, summary, timestamp），若未命中回傳 None
        """
        with self._lock:
            self._ensure_loaded()
            key = self._hash_key(text)
            entry = self._data.get(key)

            if entry is None:
                return None

            # 檢查是否過期
            if self._is_expired(entry):
                # 過期條目視為快取未命中
                del self._data[key]
                self._pending_writes += 1
                if self._pending_writes >= self._flush_every:
                    self._flush_to_disk()
                    self._pending_writes = 0
                return None

            return entry

    def put(self, text: str, result: dict) -> None:
        """將評分結果存入快取。

        Args:
            text: 輸入文本
            result: 結果字典，應包含 score, stage, summary 等欄位
        """
        with self._lock:
            self._ensure_loaded()
            key = self._hash_key(text)

            # 補上時間戳記（ISO 格式）
            entry = {
                "score": result.get("score"),
                "stage": result.get("stage"),
                "summary": result.get("summary"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._data[key] = entry
            self._pending_writes += 1

            # 累積寫入達到閾值時刷新到磁碟
            if self._pending_writes >= self._flush_every:
                self._flush_to_disk()
                self._pending_writes = 0

    def has(self, text: str) -> bool:
        """檢查文本是否已有快取結果（考慮 TTL）。

        Args:
            text: 輸入文本

        Returns:
            若快取中存在有效條目回傳 True，否則回傳 False
        """
        return self.get(text) is not None

    def clear(self) -> None:
        """清除所有快取資料，同時刪除磁碟上的快取檔案。"""
        with self._lock:
            self._data = {}
            self._pending_writes = 0
            self._loaded = True

            # 刪除磁碟上的快取檔案
            if self._cache_file.exists():
                try:
                    self._cache_file.unlink()
                except OSError:
                    pass

    def flush(self) -> None:
        """強制將記憶體中的快取資料寫入磁碟。

        用於確保所有待寫入的變更都已持久化。
        """
        with self._lock:
            self._ensure_loaded()
            self._flush_to_disk()
            self._pending_writes = 0

    def stats(self) -> dict:
        """取得快取統計資訊。

        Returns:
            包含 total_entries（總條目數）與 cache_file_size_kb（檔案大小，KB）的字典
        """
        with self._lock:
            self._ensure_loaded()
            total_entries = len(self._data)

            # 計算快取檔案大小
            if self._cache_file.exists():
                cache_file_size_kb = self._cache_file.stat().st_size / 1024
            else:
                cache_file_size_kb = 0.0

            return {
                "total_entries": total_entries,
                "cache_file_size_kb": round(cache_file_size_kb, 2),
            }
