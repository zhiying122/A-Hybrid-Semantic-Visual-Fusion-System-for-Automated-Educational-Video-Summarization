"""
系統診斷腳本 - 一次檢查所有依賴
用法：python tools/diagnose.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append(("OK", name, msg or ""))
        print(f"  ✓  {name:<30} {msg or ''}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  ✗  {name:<30} {e}")

print("\n===== KeyTake AI 系統診斷 =====\n")

# 先 import config 讓路徑設定生效
try:
    import config
except Exception:
    pass

# 1. FFmpeg
def chk_ffmpeg():
    import subprocess
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 指令失敗")
    line = r.stdout.decode(errors="ignore").split("\n")[0]
    return line[:50]
check("FFmpeg (系統)", chk_ffmpeg)

# 2. ffmpeg-python
def chk_ffmpeg_python():
    import ffmpeg
    return f"版本 {ffmpeg.__version__}" if hasattr(ffmpeg, "__version__") else "OK"
check("ffmpeg-python (套件)", chk_ffmpeg_python)

# 3. OpenCV
def chk_cv2():
    import cv2
    return f"版本 {cv2.__version__}"
check("OpenCV (cv2)", chk_cv2)

# 4. NumPy
def chk_numpy():
    import numpy as np
    return f"版本 {np.__version__}"
check("NumPy", chk_numpy)

# 5. MediaPipe
def chk_mediapipe():
    import mediapipe as mp
    v = mp.__version__
    # 測試 Tasks API
    try:
        from mediapipe.tasks import python as mp_python
        return f"版本 {v} (Tasks API OK)"
    except Exception:
        pass
    # 測試 Legacy API
    try:
        _ = mp.solutions.hands
        return f"版本 {v} (Legacy API OK)"
    except Exception:
        return f"版本 {v} (警告：兩種 API 都無法使用)"
check("MediaPipe", chk_mediapipe)

# 6. Whisper
def chk_whisper():
    import whisper
    return "套件存在（模型需第一次執行時下載）"
check("OpenAI Whisper", chk_whisper)

# 7. Sentence-Transformers
def chk_sbert():
    from sentence_transformers import SentenceTransformer
    return "OK"
check("Sentence-Transformers", chk_sbert)

# 8. pytesseract
def chk_tesseract():
    import pytesseract
    v = pytesseract.get_tesseract_version()
    return f"版本 {v}"
check("Tesseract OCR", chk_tesseract)

# 9. FastAPI
def chk_fastapi():
    import fastapi
    return f"版本 {fastapi.__version__}"
check("FastAPI", chk_fastapi)

# 10. scikit-learn
def chk_sklearn():
    import sklearn
    return f"版本 {sklearn.__version__}"
check("scikit-learn", chk_sklearn)

# 11. config 載入
def chk_config():
    import config
    return f"WHISPER_MODEL={config.WHISPER_MODEL}, ALPHA={config.ALPHA}"
check("config.py", chk_config)

# 12. Pipeline 模組串接
def chk_pipeline():
    from src.semantic.scorer import SemanticScorer
    from src.fusion.adaptive_fusion import fuse_scores
    from src.fusion.evaluator import compute_recall
    return "所有核心模組可 import"
check("Pipeline 模組串接", chk_pipeline)

# 摘要
ok = sum(1 for r in results if r[0] == "OK")
fail = sum(1 for r in results if r[0] == "FAIL")
print(f"\n===== 結果：{ok} 通過 / {fail} 失敗 =====")

if fail > 0:
    print("\n需要修復的項目：")
    for status, name, msg in results:
        if status == "FAIL":
            print(f"  - {name}: {msg}")
    print()
else:
    print("\n所有依賴正常，系統可以運行！\n")
