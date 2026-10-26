"""FastAPI endpoints for KeyTake AI video processing."""

import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = Path(os.getenv("KEYTAKE_UPLOAD_DIR", PROJECT_ROOT / "tmp" / "uploads")).resolve()
OUTPUT_DIR = Path(os.getenv("KEYTAKE_OUTPUT_DIR", PROJECT_ROOT / "tmp" / "outputs")).resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = int(os.getenv("KEYTAKE_MAX_UPLOAD_MB", "500")) * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MATERIAL_FILES = {"study_notes.md", "mind_map.json", "flashcards.json", "chapters.json"}
TASK_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="KeyTake AI",
    description="自動化教學精華擷取平台",
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _validate_task_id(task_id: str) -> str:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise HTTPException(status_code=400, detail="task_id 格式不正確")
    return task_id


def _task_output_dir(task_id: str) -> Path:
    return OUTPUT_DIR / _validate_task_id(task_id)


def _get_task_store():
    from src.platform.task_store import task_store
    return task_store


@app.get("/")
def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.is_file():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "KeyTake AI API is running"}


@app.post("/upload", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="請選擇影片檔案")

    original_name = Path(file.filename.replace("\\", "/")).name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支援的影片副檔名：{extension or '無副檔名'}",
        )

    task_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / f"{task_id}{extension}"
    total_bytes = 0

    try:
        with save_path.open("wb") as destination:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"影片大小超過限制（{MAX_UPLOAD_BYTES // (1024 * 1024)} MB）",
                    )
                destination.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="上傳檔案是空的")

        try:
            from src.platform.celery_app import process_video_task

            # Use one ID for both the API task and Celery task so status lookup is consistent.
            celery_task = process_video_task.apply_async(
                args=[task_id, str(save_path)],
                task_id=task_id,
            )
            return {
                "task_id": task_id,
                "celery_task_id": celery_task.id,
                "status": "queued",
                "mode": "celery",
            }
        except Exception as exc:
            # Fallback is opt-in to avoid accidentally running a task twice when enqueue
            # succeeded remotely but the client connection failed before confirmation.
            if os.getenv("KEYTAKE_SYNC_FALLBACK", "0").lower() not in {"1", "true", "yes"}:
                raise HTTPException(
                    status_code=503,
                    detail="目前無法排入背景任務，請確認 Redis 與 Celery worker 狀態",
                ) from exc

            task_store = _get_task_store()
            task_store[task_id] = {"status": "queued", "mode": "sync_fallback"}
            thread = threading.Thread(
                target=_run_sync,
                args=(task_id, str(save_path)),
                daemon=True,
                name=f"keytake-{task_id[:8]}",
            )
            thread.start()
            return {
                "task_id": task_id,
                "status": "queued",
                "mode": "sync_fallback",
            }
    except Exception:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task_id = _validate_task_id(task_id)
    task_store = _get_task_store()
    local_task = task_store.get(task_id)

    if isinstance(local_task, dict) and local_task.get("mode") == "sync_fallback":
        return local_task

    try:
        from celery.result import AsyncResult
        from src.platform.celery_app import process_video_task

        result = AsyncResult(task_id, app=process_video_task.app)
        state = result.state

        if state == "SUCCESS":
            return {"status": "done", "result": result.result}
        if state == "FAILURE":
            return {"status": "error", "error": str(result.result)}
        if state == "REVOKED":
            return {"status": "error", "error": "任務已取消"}
        if state in {"STARTED", "RETRY"}:
            return {"status": "processing"}
        if state == "PENDING":
            if local_task:
                return local_task
            return {"status": "queued"}
    except Exception:
        if local_task:
            return local_task
        return JSONResponse(
            status_code=503,
            content={"error": "目前無法取得任務狀態，請稍後再試"},
        )

    if local_task:
        return local_task
    raise HTTPException(status_code=404, detail="找不到此任務")


@app.get("/download/{task_id}")
def download_summary(task_id: str):
    output_path = _task_output_dir(task_id) / "summary.mp4"
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="摘要影片尚未生成")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"keytake_summary_{task_id[:8]}.mp4",
    )


@app.get("/materials/{task_id}/{filename}")
def get_study_material(task_id: str, filename: str):
    if filename not in MATERIAL_FILES:
        raise HTTPException(status_code=400, detail=f"不支援的檔案：{filename}")

    file_path = _task_output_dir(task_id) / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"檔案尚未生成：{filename}")

    if filename.endswith(".json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"無法讀取素材檔案：{filename}") from exc
        return JSONResponse(content=data)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"無法讀取素材檔案：{filename}") from exc
    return JSONResponse(content={"content": content, "filename": filename})


def _run_sync(task_id: str, video_path: str) -> None:
    """Run the processing pipeline in a development-only background thread."""
    task_store = _get_task_store()
    from src.output.video_exporter import export_index, export_summary_video

    steps = [
        ("影音前處理", 0.10),
        ("語音轉錄與語意評分", 0.50),
        ("視覺特徵提取", 0.30),
        ("多模態融合與剪輯", 0.10),
    ]
    started_at = time.time()

    def update_progress(step_idx: int, message: str = "") -> None:
        bounded_step = max(0, min(int(step_idx), len(steps)))
        completed_ratio = sum(ratio for _, ratio in steps[:bounded_step])
        elapsed = time.time() - started_at
        estimated_total = elapsed / completed_ratio if completed_ratio > 0 else 0
        remaining = max(0, estimated_total - elapsed) if estimated_total else 0
        eta = f"{int(remaining // 60)}分{int(remaining % 60)}秒" if completed_ratio else "計算中..."

        task_store[task_id] = {
            "status": "processing",
            "mode": "sync_fallback",
            "step": bounded_step,
            "step_name": steps[bounded_step][0] if bounded_step < len(steps) else "完成",
            "step_total": len(steps),
            "progress_pct": min(99, int(completed_ratio * 100)),
            "eta": eta,
            "message": message,
        }

    task_store[task_id] = {"status": "processing", "mode": "sync_fallback"}
    try:
        from main import run_pipeline

        update_progress(0, "開始處理影片...")
        output_dir = _task_output_dir(task_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = run_pipeline(
            video_path,
            output_dir=str(output_dir),
            progress_callback=update_progress,
        )

        segments = result.get("segments", [])
        update_progress(3, "輸出摘要影片...")
        if segments:
            export_summary_video(video_path, segments, str(output_dir / "summary.mp4"))

        export_index(
            segments,
            str(output_dir / "index.json"),
            course_summary=result.get("course_summary"),
        )

        index_data = {"segments": [], "course_summary": {}}
        index_path = output_dir / "index.json"
        if index_path.is_file():
            raw_index = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(raw_index, list):
                index_data = {"segments": raw_index, "course_summary": {}}
            elif isinstance(raw_index, dict):
                index_data = raw_index

        task_store[task_id] = {
            "status": "done",
            "mode": "sync_fallback",
            "progress_pct": 100,
            "step": len(steps),
            "step_name": "完成",
            "result": {
                "segments": index_data.get("segments", []),
                "course_summary": index_data.get("course_summary", {}),
                "original_duration": result.get("original_duration"),
                "summary_duration": result.get("summary_duration"),
                "time_saving_rate": result.get("time_saving_rate"),
            },
        }
    except Exception as exc:
        traceback.print_exc()
        task_store[task_id] = {
            "status": "error",
            "mode": "sync_fallback",
            "error": str(exc),
        }
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            pass
