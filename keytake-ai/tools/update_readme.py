"""
README 自動更新工具
掃描專案目錄結構，自動更新 README.md 中的系統架構樹狀圖。
每當新增、刪除或搬移原始碼檔案時執行此腳本即可保持文件同步。

用法：
  cd keytake-ai
  python tools/update_readme.py
"""

import os
import re
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"

# 要掃描的目錄與排除規則
SCAN_DIRS = ["src", "data", "tools", "tests", "weights"]
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".hypothesis",
    ".pytest_cache",
    "node_modules",
    ".git",
    "tmp",
    "output",
    ".env",
    "example",
}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".webm"}
EXCLUDE_FILES = {".gitkeep", "_test_flow.py"}

# 檔案用途說明對照表（從模組 docstring 或已知功能對應）
FILE_DESCRIPTIONS = {
    # main & config
    "main.py": "主流程 pipeline",
    "config.py": "系統參數與閾值",
    "requirements.txt": "Python 依賴",
    ".env.example": "環境變數範本",
    # src/preprocessing
    "preprocessor.py": "FFmpeg 轉碼 + 動態 SNR 降噪",
    # src/semantic
    "transcriber.py": "Whisper 語音轉錄（本地/遠端雙模式）",
    "scorer.py": "三軌語意評分（LLM / SBERT / TF-IDF）",
    "prosodic_analyzer.py": "韻律特徵提取（語速/音量/音高/停頓）",
    "llm_cache.py": "LLM 呼叫快取（避免重複計費）",
    # src/visual
    "hand_tracker.py": "MediaPipe 手部追蹤 + 卡爾曼濾波",
    "gesture_classifier.py": "GRU 手勢意圖分類器（5 類）",
    "visual_scorer.py": "OCR→SBERT 視覺分數 + IoU 降級",
    "text_detector.py": "Tesseract 文字框偵測",
    "sbert_calculator.py": "OCR 與 ASR 語意相似度",
    "clip_scorer.py": "CLIP 視覺-文字跨模態對齊",
    "visual_reliability.py": "視覺可靠度估測器（四維度）",
    "srgan.py": "SRGAN 超解析度（預設 bicubic 降級）",
    # src/fusion
    "adaptive_fusion.py": "自適應置信度加權融合 + Grid Search + LOOCV",
    "mlp_fusion.py": "MLP 可學習融合模型（選用）",
    "evaluator.py": "Recall / Precision / F1 / FAR / BERTScore / TSR",
    # src/output
    "video_exporter.py": "FFmpeg 影片剪輯 + 索引匯出",
    "study_materials.py": "學習素材產生（筆記/心智圖/閃卡）",
    # src/platform
    "api.py": "FastAPI 後端（含 Celery 降級為 threading）",
    "celery_app.py": "Celery 非同步任務定義",
    "task_store.py": "任務狀態暫存（無 Redis 時用）",
    "index.html": "前端介面（三色時間軸 + 進度條）",
    # data
    "corpus.py": "教學提示語料庫（約 60 句引導語）",
    "gesture_dataset.json": "手勢訓練資料",
    "gesture_dataset_augmented.json": "手勢增強資料集",
    "gesture_dataset_real.json": "真實場域手勢資料",
    # tools
    "annotator.py": "Ground Truth 標註（含 Cohen's Kappa）",
    "export_transcript.py": "匯出 Whisper 逐字稿",
    "run_grid_search.py": "Grid Search 找最佳 α/β",
    "run_loocv.py": "Leave-One-Out 交叉驗證",
    "run_batch_eval.py": "批次評估（多部影片）",
    "run_domain_eval.py": "跨場域評估（輸出 Markdown 報告）",
    "run_demo_benchmark.py": "模擬初步實驗結果",
    "collect_gesture_data.py": "手勢軌跡資料蒐集",
    "train_gesture_classifier.py": "GRU 手勢分類器訓練",
    "train_mlp_fusion.py": "MLP 融合模型訓練",
    "run_gesture_pipeline.py": "手勢辨識獨立測試",
    "demo.py": "快速示範腳本",
    "diagnose.py": "環境診斷工具",
    "update_readme.py": "README 自動更新",
    # tests
    "smoke_test.py": "不需真實影片的 pipeline 驗證",
    "test_evaluator.py": "評估指標（property-based）",
    "test_adaptive_fusion.py": "融合模組（property-based）",
    "test_visual_scorer.py": "視覺評分模組",
    "test_visual_reliability.py": "視覺可靠度估測器",
    "test_text_detector.py": "文字偵測模組",
    "test_sbert_calculator.py": "SBERT 計算器",
    "test_semantic_scorer.py": "語意評分模組",
    "test_gesture_classifier.py": "手勢分類器",
    # weights
    "gesture_classifier.pth": "手勢分類器權重",
    "mlp_fusion.pth": "MLP 融合模型權重",
}

# 目錄用途說明
DIR_DESCRIPTIONS = {
    "videos": "測試影片（blackboard / slides / challenging）",
    "annotations": "Ground Truth 標註",
    "weights": "預訓練模型權重",
}


def should_include(path: Path) -> bool:
    """判斷檔案是否應列入架構圖"""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return False
    if path.suffix in EXCLUDE_EXTENSIONS:
        return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.name == "__init__.py":
        return False
    return True


def build_tree(root: Path) -> list[str]:
    """
    產生樹狀圖文字行，模仿 tree 指令輸出。
    只列出 src/data/tools/tests/weights 底下的內容，
    頂層另外手動加入 main.py / config.py 等。
    """
    lines = ["keytake-ai/"]

    # 頂層檔案
    top_files = ["main.py", "config.py", "requirements.txt", ".env.example"]
    for f in top_files:
        fp = root / f
        if fp.exists():
            desc = FILE_DESCRIPTIONS.get(f, "")
            suffix = f"# {desc}" if desc else ""
            lines.append(f"├── {f:<40}{suffix}")

    lines.append("│")

    # 掃描各子目錄
    for i, dir_name in enumerate(SCAN_DIRS):
        dir_path = root / dir_name
        if not dir_path.exists():
            continue

        is_last_dir = i == len(SCAN_DIRS) - 1
        prefix = "└── " if is_last_dir else "├── "
        child_prefix = "    " if is_last_dir else "│   "

        lines.append(f"{prefix}{dir_name}/")

        # 收集此目錄下所有檔案與子目錄
        entries = _collect_entries(dir_path, root / dir_name)
        for j, (rel_path, is_dir, depth) in enumerate(entries):
            is_last = j == len(entries) - 1
            indent = child_prefix + "│   " * (depth - 1)
            connector = "└── " if is_last or _is_last_at_depth(entries, j, depth) else "├── "

            name = rel_path.name
            if is_dir:
                desc = DIR_DESCRIPTIONS.get(name, "")
                suffix = f"# {desc}" if desc else ""
                lines.append(f"{indent}{connector}{name + '/':<36}{suffix}")
            else:
                desc = FILE_DESCRIPTIONS.get(name, "")
                suffix = f"# {desc}" if desc else ""
                lines.append(f"{indent}{connector}{name:<36}{suffix}")

        if not is_last_dir:
            lines.append("│")

    return lines


def _collect_entries(base: Path, rel_base: Path) -> list[tuple[Path, bool, int]]:
    """遞迴收集目錄中的檔案（排除 __pycache__ 等）"""
    results = []
    _walk(base, rel_base, results, depth=1)
    return results


def _walk(current: Path, rel_base: Path, results: list, depth: int):
    """深度優先走訪"""
    if not current.is_dir():
        return

    children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for child in children:
        if not should_include(child.relative_to(rel_base.parent)):
            continue

        if child.is_dir():
            # 只列出含有有效檔案的目錄
            sub_files = [f for f in child.rglob("*") if f.is_file() and should_include(f.relative_to(rel_base.parent))]
            if sub_files:
                results.append((child, True, depth))
                _walk(child, rel_base, results, depth + 1)
        else:
            results.append((child, False, depth))


def _is_last_at_depth(entries, idx, depth):
    """判斷某項目是否為同深度的最後一個"""
    for k in range(idx + 1, len(entries)):
        if entries[k][2] == depth:
            return False
        if entries[k][2] < depth:
            return True
    return True


def generate_tree_block(root: Path) -> str:
    """產生完整的 markdown 程式碼區塊"""
    lines = build_tree(root)
    return "```\n" + "\n".join(lines) + "\n```"


def update_readme(readme_path: Path, tree_block: str):
    """將 README 中的系統架構區塊替換為最新內容"""
    if not readme_path.exists():
        print(f"[錯誤] 找不到 {readme_path}")
        return False

    content = readme_path.read_text(encoding="utf-8")

    # 匹配 ## 系統架構 後面的 ``` ... ``` 區塊
    pattern = r"(## 系統架構\s*\n\s*\n)```[\s\S]*?```"
    replacement = rf"\g<1>{tree_block}"

    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        print("[警告] 找不到「## 系統架構」區塊，跳過更新")
        return False

    readme_path.write_text(new_content, encoding="utf-8")
    print(f"[完成] 已更新 {readme_path.name} 的系統架構區塊")
    return True


def main():
    tree_block = generate_tree_block(PROJECT_ROOT)
    success = update_readme(README_PATH, tree_block)
    if success:
        print("README.md 系統架構已同步至最新目錄結構。")
    else:
        print("更新失敗，請確認 README.md 存在且包含「## 系統架構」區段。")


if __name__ == "__main__":
    main()
