"""
check_env.py — 檢查 KeyTake AI 執行環境是否完備
逐一嘗試 import 所有必要與選填套件，並檢查外部工具與環境變數。
"""
import importlib
import os
import shutil
import sys

REQUIRED = [
    ("whisper", "openai-whisper", "語音轉錄"),
    ("sentence_transformers", "sentence-transformers", "SBERT 語意"),
    ("torch", "torch", "深度學習框架"),
    ("torchvision", "torchvision", "視覺模型"),
    ("bert_score", "bert-score", "評估指標"),
    ("cv2", "opencv-python", "影像處理"),
    ("mediapipe", "mediapipe", "手部追蹤"),
    ("pytesseract", "pytesseract", "OCR"),
    ("ffmpeg", "ffmpeg-python", "FFmpeg 包裝"),
    ("sklearn", "scikit-learn", "TF-IDF"),
    ("jieba", "jieba", "中文斷詞"),
    ("numpy", "numpy", "數值運算"),
    ("scipy", "scipy", "科學運算"),
    ("fastapi", "fastapi", "Web API"),
    ("uvicorn", "uvicorn", "ASGI 伺服器"),
    ("multipart", "python-multipart", "檔案上傳"),
    ("pydantic", "pydantic", "資料驗證"),
    ("celery", "celery", "任務佇列"),
    ("redis", "redis", "Redis 客戶端"),
    ("PIL", "Pillow", "影像格式轉換"),
]

OPTIONAL = [
    ("librosa", "librosa", "韻律特徵（缺→韻律固定 0.5）"),
    ("transformers", "transformers", "CLIP 圖文對齊（缺→跳過 CLIP）"),
    ("groq", "groq", "Groq LLM 後端（缺→改用其他後端）"),
    ("openai", "openai", "OpenAI API"),
    ("hypothesis", "hypothesis", "property-based 測試"),
    ("pytest", "pytest", "測試框架"),
]


def check(mods, label):
    print(f"\n=== {label} ===")
    ok, missing = [], []
    for mod, pkg, desc in mods:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK]  {pkg:<24} {ver:<12} {desc}")
            ok.append(pkg)
        except Exception as e:
            print(f"  [X ]  {pkg:<24} {'MISSING':<12} {desc}  ({type(e).__name__})")
            missing.append(pkg)
    return ok, missing


def main():
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    _, miss_req = check(REQUIRED, "必要套件")
    _, miss_opt = check(OPTIONAL, "選填套件")

    print("\n=== 外部工具 ===")
    for tool in ("ffmpeg", "tesseract"):
        path = shutil.which(tool)
        if not path and tool == "tesseract":
            for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                if os.path.isfile(c):
                    path = c
                    break
        print(f"  [{'OK' if path else 'X '}]  {tool:<12} {path or 'NOT FOUND'}")

    print("\n=== 環境變數 (.env) ===")
    for key in ("SEMANTIC_SCORER_MODE", "LLM_BACKEND", "WHISPER_MODE", "OPENAI_API_KEY", "GROQ_API_KEY"):
        val = os.environ.get(key)
        if key.endswith("_KEY") and val:
            val = "已設定（不顯示值）"
        print(f"  {key:<22} = {val or '(未設定)'}")

    print("\n" + "=" * 50)
    if miss_req:
        print(f"[結果] 缺少必要套件：{miss_req}")
        sys.exit(1)
    print("[結果] 所有必要套件皆可用 ✓")
    if miss_opt:
        print(f"[提示] 選填套件未裝（系統會自動降級）：{miss_opt}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    main()
