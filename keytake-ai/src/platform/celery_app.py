"""
Celery 任務佇列設定
對應計畫書 4.3：非同步處理架構，高算力模組後端排程執行

需要：
  pip install celery redis
  並啟動 Redis server（預設 localhost:6379）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from celery import Celery

# Redis 作為 broker 與 result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "keytake",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.platform.celery_app"]
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
)


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, task_id: str, video_path: str) -> dict:
    """
    Celery 非同步任務：執行完整摘要 pipeline
    SRGAN 超解析模組僅在偵測到模糊手寫字時觸發（計畫書 4.3）
    """
    from main import run_pipeline
    from src.output.video_exporter import export_summary_video, export_index

    out_dir = f"tmp/outputs/{task_id}"
    os.makedirs(out_dir, exist_ok=True)

    self.update_state(state="STARTED", meta={"progress": 10, "step": "pipeline"})
    result = run_pipeline(video_path, output_dir=out_dir)

    self.update_state(state="STARTED", meta={"progress": 80, "step": "exporting"})
    summary_path = os.path.join(out_dir, "summary.mp4")
    export_summary_video(video_path, result["segments"], summary_path)

    index_path = os.path.join(out_dir, "index.json")
    export_index(result["segments"], index_path)

    # 保留原始影片，避免使用者檔案被意外刪除
    # 僅清理系統暫存的副本（從 UPLOAD_DIR 複製的檔案）
    upload_dir = os.path.join("tmp", "uploads")
    if os.path.exists(video_path) and os.path.abspath(video_path).startswith(os.path.abspath(upload_dir)):
        os.remove(video_path)

    import json
    return {
        "segments": json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else [],
        "original_duration": result["original_duration"],
        "summary_duration": result["summary_duration"],
        "time_saving_rate": result["time_saving_rate"],
    }
