"""Celery 任務設定：使用 Redis 執行 KeyTake AI 影片摘要。"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from celery import Celery

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # Load project .env settings before reading the Redis URL.

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_DIR = Path(
    os.getenv("KEYTAKE_UPLOAD_DIR", str(PROJECT_ROOT / "tmp" / "uploads"))
).resolve()
OUTPUT_DIR = Path(
    os.getenv("KEYTAKE_OUTPUT_DIR", str(PROJECT_ROOT / "tmp" / "outputs"))
).resolve()

celery_app = Celery(
    "keytake",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Taipei",
    enable_utc=False,
    task_track_started=True,
    task_soft_time_limit=1800,
    task_time_limit=2100,
    result_expires=86400,
    broker_connection_timeout=2,
    redis_socket_connect_timeout=1,
    redis_socket_timeout=2,
    result_backend_max_retries=3,
    result_backend_base_sleep_between_retries_ms=100,
    result_backend_max_sleep_between_retries_ms=500,
)


def _is_inside(path: Path, directory: Path) -> bool:
    """確認路徑是否位於指定目錄內，避免誤刪其他檔案。"""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, task_id: str, video_path: str) -> dict[str, Any]:
    """執行影片摘要流程並回傳摘要索引與統計資料。"""
    from main import run_pipeline
    from src.output.video_exporter_optimized import (
        export_index,
        export_summary_video,
        probe_video_duration,
    )

    source_path = Path(video_path).resolve()
    output_dir = OUTPUT_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        self.update_state(
            state="STARTED",
            meta={"progress": 5, "step": "準備影片", "task_id": task_id},
        )

        result = run_pipeline(str(source_path), output_dir=str(output_dir))

        self.update_state(
            state="STARTED",
            meta={"progress": 75, "step": "輸出摘要影片", "task_id": task_id},
        )

        segments = result.get("segments", [])
        summary_path = output_dir / "summary.mp4"
        timeline_path = output_dir / "timeline.json"
        timeline = []
        summary_duration = result.get("summary_duration")
        if segments:
            export_summary_video(
                str(source_path),
                segments,
                str(summary_path),
                timeline_output_path=str(timeline_path),
            )
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            summary_duration = probe_video_duration(str(summary_path))

        index_path = output_dir / "index.json"
        export_index(
            segments,
            str(index_path),
            course_summary=result.get("course_summary"),
        )

        with index_path.open("r", encoding="utf-8") as file:
            index_data = json.load(file)

        self.update_state(
            state="STARTED",
            meta={"progress": 95, "step": "整理輸出資料", "task_id": task_id},
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "step": "完成",
            "segments": index_data.get("segments", []),
            "course_summary": index_data.get("course_summary", {}),
            "original_duration": result.get("original_duration"),
            "summary_duration": summary_duration,
            "time_saving_rate": (
                1.0 - summary_duration / result["original_duration"]
                if summary_duration
                and result.get("original_duration", 0) > 0
                else result.get("time_saving_rate")
            ),
            "timeline": timeline,
            "summary_video": str(summary_path) if summary_path.exists() else None,
        }
    finally:
        # 僅清除專案上傳暫存目錄內的檔案，不碰其他來源檔案。
        if source_path.is_file() and _is_inside(source_path, UPLOAD_DIR):
            try:
                source_path.unlink()
            except OSError:
                # 清理失敗不應覆蓋原本的任務例外；可由定期清理程序處理。
                pass
