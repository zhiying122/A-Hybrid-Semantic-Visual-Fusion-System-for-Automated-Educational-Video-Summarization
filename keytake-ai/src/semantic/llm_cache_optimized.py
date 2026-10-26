"""LLM 評分結果的 JSON 快取，支援 TTL、延遲載入與批次寫入。"""

import hashlib
import json
import os
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LLMCache:
    """以輸入文本的 SHA-256 雜湊值快取 LLM 評分結果。

    採用延遲載入（首次 get/put 才讀磁碟）與批次寫入（累積 flush_every 筆再刷盤），
    以減少重複 I/O。寫盤使用原子替換（write-to-temp + os.replace）確保完整性。
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        max_age_days: float | None = None,
        flush_every: int = 10,
    ) -> None:
        if max_age_days is not None and max_age_days < 0:
            raise ValueError("max_age_days 不可小於 0")
        if flush_every < 1:
            raise ValueError("flush_every 必須至少為 1")

        self._cache_dir = Path(cache_dir)
        self._cache_file = self._cache_dir / "llm_scores.json"
        self._max_age_days = max_age_days
        self._flush_every = flush_every
        self._pending_writes = 0
        self._lock = threading.RLock()
        self._loaded = False
        self._data: dict[str, dict[str, Any]] = {}

        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            with self._cache_file.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                self._data = {
                    key: value
                    for key, value in loaded.items()
                    if isinstance(key, str) and isinstance(value, dict)
                }
            else:
                warnings.warn(
                    f"快取格式不正確，將使用空快取：{self._cache_file}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._data = {}
        except FileNotFoundError:
            self._data = {}
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(
                f"讀取快取失敗，將使用空快取：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._data = {}

        self._loaded = True

    @staticmethod
    def _hash_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        if self._max_age_days is None:
            return False
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
            return age_seconds > self._max_age_days * 86400
        except (KeyError, TypeError, ValueError):
            return True

    def _flush_to_disk(self) -> bool:
        """原子更新快取檔（write-to-temp + os.replace）；成功回傳 True。"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # 帶 PID+TID 的 temp 名稱避免多 worker 衝突
        temp_path = self._cache_file.with_name(
            f".{self._cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )

        try:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(self._data, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self._cache_file)
            return True
        except OSError as exc:
            warnings.warn(
                f"寫入快取失敗，待寫入資料仍保留在記憶體：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def get(self, text: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_loaded()
            key = self._hash_key(text)
            entry = self._data.get(key)

            if entry is None:
                return None
            if self._is_expired(entry):
                del self._data[key]
                self._pending_writes += 1
                if self._pending_writes >= self._flush_every:
                    if self._flush_to_disk():
                        self._pending_writes = 0
                return None

            return dict(entry)

    def put(self, text: str, result: dict[str, Any]) -> None:
        if not isinstance(text, str):
            raise TypeError("text 必須是字串")
        if not isinstance(result, dict):
            raise TypeError("result 必須是字典")

        entry = {
            "score": result.get("score"),
            "stage": result.get("stage"),
            "summary": result.get("summary"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._ensure_loaded()
            self._data[self._hash_key(text)] = entry
            self._pending_writes += 1

            if self._pending_writes >= self._flush_every:
                if self._flush_to_disk():
                    self._pending_writes = 0

    def has(self, text: str) -> bool:
        return self.get(text) is not None

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._pending_writes = 0
            self._loaded = True
            try:
                self._cache_file.unlink(missing_ok=True)
            except OSError as exc:
                warnings.warn(f"刪除快取檔失敗：{exc}", RuntimeWarning, stacklevel=2)

    def flush(self) -> bool:
        """強制將待寫入資料刷入磁碟；回傳是否成功。"""
        with self._lock:
            self._ensure_loaded()
            if self._flush_to_disk():
                self._pending_writes = 0
                return True
            return False

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            self._ensure_loaded()
            try:
                size_kb = (
                    self._cache_file.stat().st_size / 1024
                    if self._cache_file.exists()
                    else 0.0
                )
            except OSError:
                size_kb = 0.0

            return {
                "total_entries": len(self._data),
                "pending_writes": self._pending_writes,
                "cache_file_size_kb": round(size_kb, 2),
            }
