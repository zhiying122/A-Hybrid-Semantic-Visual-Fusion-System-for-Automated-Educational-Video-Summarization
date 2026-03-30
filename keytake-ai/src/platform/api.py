"""
4.3 系統平台建置：FastAPI 後端
- 非同步任務佇列（Celery + Redis）處理長影片
- 提供影片上傳、狀態查詢、摘要下載 API
- 靜態前端介面服務

啟動方式：
  # 1. 啟動 Redis（需先安裝）
  #    Windows: 下載 Redis for Windows 或用 WSL
  # 2. 啟動 Celery worker
  #    cd keytake-ai && celery -A src.platform.celery_app worker --loglevel=info
  # 3. 啟動 FastAPI
  #    cd keytake-ai && uvicorn src.platform.api:app --reload
"""

import os
import sys
import uuid

# 確保從 keytake-ai/ 根目錄可以 import main 與其他模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="KeyTake AI", description="自動化教學精華擷取平台")

# 靜態前端
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

UPLOAD_DIR = "tmp/uploads"
OUTPUT_DIR = "tmp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def index():
    """提供前端介面"""
    html_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "KeyTake AI API is running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    上傳教學影片，交由 Celery 非同步處理
    回傳 task_id 供後續查詢
    """
    task_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 嘗試使用 Celery，失敗則降級為 BackgroundTasks
    try:
        from src.platform.celery_app import process_video_task
        celery_task = process_video_task.delay(task_id, save_path)
        return {"task_id": task_id, "celery_task_id": celery_task.id, "status": "queued"}
    except Exception:
        # Celery 未啟動時降級為同步處理（開發用）
        from src.platform.task_store import task_store
        task_store[task_id] = {"status": "queued"}
        import threading
        threading.Thread(target=_run_sync, args=(task_id, save_path), daemon=True).start()
        return {"task_id": task_id, "status": "queued", "mode": "sync_fallback"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """查詢任務處理狀態"""
    # 先查 Celery 結果後端
    try:
        from src.platform.celery_app import process_video_task
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        if result.state == "SUCCESS":
            return {"status": "done", "result": result.result}
        elif result.state == "FAILURE":
            return {"status": "error", "error": str(result.result)}
        elif result.state in ("PENDING", "STARTED"):
            return {"status": "processing"}
    except Exception:
        pass

    # 降級：查本地 task_store
    from src.platform.task_store import task_store
    if task_id not in task_store:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return task_store[task_id]


@app.get("/download/{task_id}")
def download_summary(task_id: str):
    """下載摘要影片"""
    output_path = os.path.join(OUTPUT_DIR, task_id, "summary.mp4")
    if not os.path.exists(output_path):
        return JSONResponse(status_code=404, content={"error": "摘要影片尚未生成"})
    return FileResponse(output_path, media_type="video/mp4",
                        filename=f"keytake_summary_{task_id[:8]}.mp4")


def _run_sync(task_id: str, video_path: str):
    """降級同步處理（Celery 未啟動時使用）"""
    from src.platform.task_store import task_store
    from src.output.video_exporter import export_summary_video, export_index
    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    # 步驟定義：(步驟名, 預估佔比)
    STEPS = [
        ("影音前處理", 0.10),
        ("語音轉錄與語意評分", 0.50),
        ("視覺特徵提取", 0.30),
        ("多模態融合與剪輯", 0.10),
    ]
    start_time = time.time()

    def update_progress(step_idx: int, message: str = ""):
        elapsed = time.time() - start_time
        # 預估總時間（根據已完成比例推算）
        done_ratio = sum(r for _, r in STEPS[:step_idx])
        if done_ratio > 0:
            estimated_total = elapsed / done_ratio
            remaining = max(0, estimated_total - elapsed)
            eta = f"{int(remaining // 60)}分{int(remaining % 60)}秒"
        else:
            eta = "計算中..."

        task_store[task_id] = {
            "status": "processing",
            "step": step_idx,
            "step_name": STEPS[step_idx][0] if step_idx < len(STEPS) else "完成",
            "step_total": len(STEPS),
            "progress_pct": int(done_ratio * 100),
            "eta": eta,
            "message": message,
        }

    task_store[task_id] = {"status": "queued"}
    try:
        from main import run_pipeline

        update_progress(0, "轉換影片格式...")
        out_dir = os.path.join(OUTPUT_DIR, task_id)
        result = run_pipeline(video_path, output_dir=out_dir,
                              progress_callback=update_progress)

        # 剪輯摘要影片
        update_progress(3, "輸出摘要影片...")
        summary_path = os.path.join(out_dir, "summary.mp4")
        if result["segments"]:
            export_summary_video(video_path, result["segments"], summary_path)
        else:
            summary_path = None

        # 匯出索引
        index_path = os.path.join(out_dir, "index.json")
        export_index(result["segments"], index_path)

        task_store[task_id] = {
            "status": "done",
            "result": {
                "segments": result["segments"],
                "original_duration": result["original_duration"],
                "summary_duration": result["summary_duration"],
                "time_saving_rate": result["time_saving_rate"],
            }
        }
    except Exception as e:
        task_store[task_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
