# KeyTake AI 系統參數設定
# 對應計畫書 4.2 各步驟的可調參數

# ── 系統工具路徑（Windows 用戶若 PATH 未設定可在此指定）──────
import os as _os
import pytesseract as _pytesseract

_ffmpeg_path = r"C:\ffmpeg-8.1-essentials_build\bin"
if _os.path.exists(_ffmpeg_path):
    _os.environ["PATH"] = _ffmpeg_path + _os.pathsep + _os.environ.get("PATH", "")

_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if _os.path.exists(_tesseract_path):
    _pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# ── 步驟一：影音前處理 ──────────────────────────────
AUDIO_SNR_THRESHOLD = 10.0        # SNR 低於此值才啟用頻譜減法
OUTPUT_FORMAT = "mp4"

# ── 步驟二：語意分析 ────────────────────────────────
WHISPER_MODEL = "base"            # tiny / base / small / medium / large
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
