# KeyTake AI 系統參數設定
# 對應計畫書 4.2 各步驟的可調參數

# ── 步驟一：影音前處理 ──────────────────────────────
AUDIO_SNR_THRESHOLD = 10.0        # SNR 低於此值才啟用頻譜減法
OUTPUT_FORMAT = "mp4"

# ── 步驟二：語意分析 ────────────────────────────────
WHISPER_MODEL = "base"            # tiny / base / small / medium / large
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 支援中文
TFIDF_TOP_K = 20                  # 取前 K 個高權重關鍵詞
SBERT_SIMILARITY_THRESHOLD = 0.6  # 與提示語料庫的最低餘弦相似度

# ── 步驟三：視覺特徵提取 ────────────────────────────
HAND_VARIANCE_THRESHOLD = 50.0    # 手部位移變異數閾值（判斷滯留）
ROI_SIZE = 224                    # 感興趣區域裁切大小 (N×N)
IOU_FALLBACK_THRESHOLD = 0.3      # 降級備援：手部與文字框最低 IoU

# ── 步驟四：多模態融合 ──────────────────────────────
ALPHA = 0.5                       # 語意權重 α（Grid Search 調校）
BETA = 0.5                        # 視覺權重 β（α + β = 1）
FUSION_SCORE_THRESHOLD = 0.5      # 保留片段的最低融合分數
SLIDING_WINDOW_MIN_SEC = 10       # 語意感知滑動視窗最小保留秒數

# ── 評估目標（計畫書 4.4）───────────────────────────
TARGET_RECALL = 0.70
TARGET_FALSE_ALARM_RATE = 0.25
TARGET_BERT_SCORE = 0.70
TARGET_TIME_SAVING_RATE = 0.50
