
"""Celery 降級執行時使用的本機任務狀態暫存。"""

from typing import Any

task_store: dict[str, dict[str, Any]] = {}
