"""
Celery 非同步任務設定

以 Redis 作為 broker 與 result backend，
處理教學影片摘要 pipeline 的非同步執行。
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # Load project .env settings before reading the Redis URL.

from celery import Celery

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
    include=["src.platform.celery_app"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Taipei",
    task_track_started=True,
    # 長影片可能需要較長時間，設定 30 分鐘 soft limit
    task_soft_time_limit=1800,
    task_time_limit=2100,
    broker_connection_timeout=2,
    redis_socket_connect_timeout=1,
    redis_socket_timeout=2,
    result_backend_max_retries=3,
    result_backend_base_sleep_between_retries_ms=100,
    result_backend_max_sleep_between_retries_ms=500,
)


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, task_id: str, video_path: str) -> dict:
    """執行完整摘要 pipeline，回傳精華片段索引與統計資訊。"""
    from main import run_pipeline
    from src.output.video_exporter_optimized import (
        export_summary_video,
        export_index,
        probe_video_duration,
    )

    source_path = Path(video_path).resolve()
    out_dir = OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        self.update_state(state="STARTED", meta={"progress": 10, "step": "pipeline"})
        result = run_pipeline(str(source_path), output_dir=str(out_dir))

        self.update_state(state="STARTED", meta={"progress": 80, "step": "exporting"})
        segments = result["segments"]
        summary_path = out_dir / "summary.mp4"
        timeline_path = out_dir / "timeline.json"
        timeline = []
        summary_duration = result["summary_duration"]
        if segments:
            export_summary_video(
                str(source_path),
                segments,
                str(summary_path),
                timeline_output_path=str(timeline_path),
            )
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            summary_duration = probe_video_duration(str(summary_path))

        index_path = out_dir / "index.json"
        export_index(
            segments,
            str(index_path),
            course_summary=result.get("course_summary"),
        )

        with index_path.open("r", encoding="utf-8") as file:
            index_data = json.load(file)

        return {
            "task_id": task_id,
            "status": "completed",
            "segments": index_data.get("segments", []),
            "course_summary": index_data.get("course_summary", {}),
            "original_duration": result["original_duration"],
            "summary_duration": summary_duration,
            "time_saving_rate": (
                1.0 - summary_duration / result["original_duration"]
                if summary_duration and result["original_duration"] > 0
                else result["time_saving_rate"]
            ),
            "timeline": timeline,
            "summary_video": str(summary_path) if summary_path.is_file() else None,
        }
    finally:
        try:
            source_path.relative_to(UPLOAD_DIR)
        except ValueError:
            pass
        else:
            source_path.unlink(missing_ok=True)
