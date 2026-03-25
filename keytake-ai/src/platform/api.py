"""
4.3 系統平台建置：FastAPI 後端
- 離線批次處理 + 非同步任務佇列（Celery + Redis）
- 提供影片上傳與摘要結果查詢 API
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os

app = FastAPI(title="KeyTake AI", description="自動化教學精華擷取平台")

# 任務狀態暫存（正式環境應改用 Redis）
task_store: dict[str, dict] = {}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    上傳教學影片，非同步觸發摘要流程
    回傳 task_id 供後續查詢
    """
    task_id = str(uuid.uuid4())
    save_path = f"/tmp/{task_id}_{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    task_store[task_id] = {"status": "queued", "result": None}
    background_tasks.add_task(_run_pipeline, task_id, save_path)

    return {"task_id": task_id, "status": "queued"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """查詢任務處理狀態"""
    if task_id not in task_store:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return task_store[task_id]


async def _run_pipeline(task_id: str, video_path: str):
    """非同步執行完整摘要流程"""
    task_store[task_id]["status"] = "processing"
    try:
        from main import run_pipeline
        result = run_pipeline(video_path)
        task_store[task_id] = {"status": "done", "result": result}
    except Exception as e:
        task_store[task_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
