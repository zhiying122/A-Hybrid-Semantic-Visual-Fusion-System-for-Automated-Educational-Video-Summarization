"""
簡易任務狀態暫存（降級用，Celery 未啟動時使用）
正式環境應改用 Redis 作為後端
"""

task_store: dict[str, dict] = {}
