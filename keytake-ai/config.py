# KeyTake AI 系統參數設定
#
# 工具路徑優先順序：環境變數 > shutil.which() > 常見安裝路徑
import os as _os
import shutil as _shutil
from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

_load_dotenv(_Path(__file__).with_name(".env"))


def _find_ffmpeg() -> str | None:
    env_path = _os.environ.get("FFMPEG_PATH", "")
    if env_path and _os.path.isdir(env_path):
        return env_path
    if _shutil.which("ffmpeg"):
        return None
    candidates = [
        r"C:\ffmpeg-8.1-essentials_build\bin",
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ]
    for p in candidates:
        if _os.path.exists(_os.path.join(p, "ffmpeg")) or _os.path.exists(_os.path.join(p, "ffmpeg.exe")):
            return p
    return None


def _find_tesseract() -> str | None:
    env_path = _os.environ.get("TESSERACT_PATH", "")
    if env_path and _os.path.isfile(env_path):
        return env_path
    which_result = _shutil.which("tesseract")
    if which_result:
        return which_result
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for p in candidates:
        if _os.path.isfile(p):
            return p
    return None


_ffmpeg_dir = _find_ffmpeg()
if _ffmpeg_dir:
    _os.environ["PATH"] = _ffmpeg_dir + _os.pathsep + _os.environ.get("PATH", "")

try:
    import pytesseract as _pytesseract
    _tesseract_cmd = _find_tesseract()
    if _tesseract_cmd:
        _pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
except ImportError:
    pass

# ── 影音前處理 ──────────────────────────────────────
AUDIO_SNR_THRESHOLD = 10.0        # SNR 低於此值才啟用頻譜減法
OUTPUT_FORMAT = "mp4"

# ── 語意分析 ────────────────────────────────────────
WHISPER_MODEL = "base"            # tiny / base / small / medium / large
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_TOP_K = 20
SBERT_SIMILARITY_THRESHOLD = 0.6
SBERT_OCR_WEIGHT = 0.7            # SBERT 分數在視覺分數中的權重

# ── 視覺特徵提取 ────────────────────────────────────
TEXT_DETECTION_CONFIDENCE_THRESHOLD = 0.5
HAND_VARIANCE_THRESHOLD = 50.0    # 手部位移變異數閾值
ROI_SIZE = 224
VISUAL_SAMPLE_FPS = 1
IOU_FALLBACK_THRESHOLD = 0.3      # 降級備援：手部與文字框最低 IoU
BLUR_THRESHOLD = 100.0            # Laplacian 變異數閾值（模糊偵測）
SRGAN_SCALE = 4

# ── 多模態融合 ──────────────────────────────────────
ALPHA = 0.6                       # 目前採用的實驗基準；三部訓練影片搜尋候選 0.5/0.5 尚未正式採用
BETA = 0.4                        # 視覺權重（α + β = 1）；不代表已驗證的最佳值
FUSION_SCORE_THRESHOLD = 0.2      # 保留片段的最低融合分數
SLIDING_WINDOW_MIN_SEC = 10       # 語意感知滑動視窗最小保留秒數

# 摘要總時長上限：摘要不得超過「原片時長 × MAX_SUMMARY_RATIO」
MAX_SUMMARY_RATIO = 0.6
MIN_SUMMARY_SEC = 2.0             # 摘要最短輸出保底秒數

# ── 評估目標 ────────────────────────────────────────
TARGET_RECALL = 0.70
TARGET_FALSE_ALARM_RATE = 0.25
TARGET_BERT_SCORE = 0.70
TARGET_TIME_SAVING_RATE = 0.50

# ── 跨場域評估 ──────────────────────────────────────
DOMAIN_BLACKBOARD = "blackboard"
DOMAIN_SLIDES = "slides"
DOMAIN_CHALLENGING = "challenging"

# ── 手勢意圖分類器 ──────────────────────────────────
GESTURE_SEQ_LEN = 20              # GRU 輸入序列長度（幀數）
GESTURE_INTENT_THRESHOLD = 0.7    # 觸發視覺分析的最低意圖分數
GESTURE_MODEL_PATH = "weights/gesture_classifier.pth"

# ── LLM 語意評分（透過 .env 設定）──────────────────
# SEMANTIC_SCORER_MODE=llm / sbert / tfidf
# LLM_BACKEND=openai / groq / ollama
# OPENAI_API_KEY / GROQ_API_KEY / OLLAMA_URL

# ── 多幀視覺取樣 ────────────────────────────────────
VISUAL_FRAMES_PER_SEGMENT = 3     # 每個片段取幾幀（頭/中/尾）

# ── 韻律特徵分析 ────────────────────────────────────
PROSODIC_WEIGHT = 0.15
PROSODIC_SPEECH_RATE_WEIGHT = 0.3
PROSODIC_VOLUME_WEIGHT = 0.3
PROSODIC_PITCH_WEIGHT = 0.25
PROSODIC_PAUSE_WEIGHT = 0.15

# ── CLIP 視覺-文字對齊 ──────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_SCORE_THRESHOLD = 0.25
CLIP_WEIGHT = 0.3                 # CLIP 分數在視覺分數中的混合權重

# ── MLP 可學習融合模型 ──────────────────────────────
MLP_FUSION_MODEL_PATH = "weights/mlp_fusion_12d.pth"
MLP_FUSION_EPOCHS = 100
MLP_FUSION_LR = 0.001
USE_MLP_FUSION = False            # 12 維模型仍屬小樣本實驗；獨立測試不足以支持正式啟用
