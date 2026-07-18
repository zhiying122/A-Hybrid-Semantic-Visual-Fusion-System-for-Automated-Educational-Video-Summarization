# KeyTake AI 系統參數設定
# 對應計畫書 4.2 各步驟的可調參數

# ── 系統工具路徑（Windows 用戶若 PATH 未設定可在此指定）──────
import os as _os

_ffmpeg_path = r"C:\ffmpeg-8.1-essentials_build\bin"
if _os.path.exists(_ffmpeg_path):
    _os.environ["PATH"] = _ffmpeg_path + _os.pathsep + _os.environ.get("PATH", "")

try:
    import pytesseract as _pytesseract
    _tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _os.path.exists(_tesseract_path):
        _pytesseract.pytesseract.tesseract_cmd = _tesseract_path
except ImportError:
    pass  # pytesseract 未安裝時跳過路徑設定，視覺模組自行處理降級

# ── 步驟一：影音前處理 ──────────────────────────────
AUDIO_SNR_THRESHOLD = 10.0        # SNR 低於此值才啟用頻譜減法
OUTPUT_FORMAT = "mp4"

# ── 步驟二：語意分析 ────────────────────────────────
WHISPER_MODEL = "base"            # tiny（最快）/ base / small / medium / large
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 支援中文
TFIDF_TOP_K = 20                  # 取前 K 個高權重關鍵詞
SBERT_SIMILARITY_THRESHOLD = 0.6  # 與提示語料庫的最低餘弦相似度
SBERT_OCR_WEIGHT = 0.7            # SBERT 分數在視覺分數中的權重

# ── 步驟三：視覺特徵提取 ────────────────────────────
TEXT_DETECTION_CONFIDENCE_THRESHOLD = 0.5  # 文字偵測信心度閾值
HAND_VARIANCE_THRESHOLD = 50.0    # 手部位移變異數閾值（判斷滯留）
ROI_SIZE = 224                    # 感興趣區域裁切大小 (N×N)
VISUAL_SAMPLE_FPS = 1             # 視覺分析取樣頻率（每秒幾幀，降低可大幅加速）
IOU_FALLBACK_THRESHOLD = 0.3      # 降級備援：手部與文字框最低 IoU
BLUR_THRESHOLD = 100.0            # Laplacian 變異數閾值（模糊偵測）
SRGAN_SCALE = 4                   # 超解析度放大倍數

# ── 步驟四：多模態融合 ──────────────────────────────
ALPHA = 0.5                       # 語意權重 α（Grid Search 調校）
BETA = 0.5                        # 視覺權重 β（α + β = 1）
FUSION_SCORE_THRESHOLD = 0.2      # 保留片段的最低融合分數（降低以選取更多片段）
SLIDING_WINDOW_MIN_SEC = 10       # 語意感知滑動視窗最小保留秒數

# ── 評估目標（計畫書 4.4）───────────────────────────
TARGET_RECALL = 0.70
TARGET_FALSE_ALARM_RATE = 0.25    # 誤報率目標 < 25%
TARGET_BERT_SCORE = 0.70
TARGET_TIME_SAVING_RATE = 0.50

# ── 跨場域評估 ──────────────────────────────────────
DOMAIN_BLACKBOARD = "blackboard"      # 傳統黑板
DOMAIN_SLIDES = "slides"              # 投影片
DOMAIN_CHALLENGING = "challenging"    # 高反光/低對比

# ── v3：手勢意圖分類器 ──────────────────────────────
GESTURE_SEQ_LEN = 20              # 輸入 GRU 的軌跡序列長度（幀數）
GESTURE_INTENT_THRESHOLD = 0.7    # 觸發視覺分析的最低意圖分數
GESTURE_MODEL_PATH = "weights/gesture_classifier.pth"

# ── v3：LLM 語意評分 ────────────────────────────────
# 透過 .env 設定：SEMANTIC_SCORER_MODE=llm / sbert / tfidf
# LLM_BACKEND=openai / groq / ollama
# OPENAI_API_KEY / GROQ_API_KEY / OLLAMA_URL

# ── v3：多幀視覺取樣 ────────────────────────────────
VISUAL_FRAMES_PER_SEGMENT = 3     # 每個片段取幾幀（頭/中/尾）

# ── v5：韻律特徵分析 ────────────────────────────────
PROSODIC_WEIGHT = 0.15            # 韻律分數在融合中的權重
PROSODIC_SPEECH_RATE_WEIGHT = 0.3 # 語速子特徵權重
PROSODIC_VOLUME_WEIGHT = 0.3      # 音量子特徵權重
PROSODIC_PITCH_WEIGHT = 0.25      # 音高變化子特徵權重
PROSODIC_PAUSE_WEIGHT = 0.15      # 停頓比例子特徵權重

# ── v5：CLIP 視覺-文字對齊 ──────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_SCORE_THRESHOLD = 0.25       # CLIP 分數閾值（低於此值不採用）
CLIP_WEIGHT = 0.3                 # CLIP 分數在視覺分數中的混合權重

# ── v5：MLP 融合模型 ────────────────────────────────
MLP_FUSION_MODEL_PATH = "weights/mlp_fusion.pth"
MLP_FUSION_EPOCHS = 100           # 訓練輪數
MLP_FUSION_LR = 0.001             # 學習率
USE_MLP_FUSION = False            # 是否啟用 MLP 融合（需先訓練模型）
