"""Canonical FastAPI import path, retained for compatibility with existing commands."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.keytake_api_optimized import app

__all__ = ["app"]
