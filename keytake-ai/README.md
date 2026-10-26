# KeyTake AI — 程式碼說明

本目錄包含 KeyTake AI 系統的完整程式碼。
研究背景、系統架構與實驗結果請參閱[專案根目錄 README](../README.md)。

---

## 環境需求

- Python 3.10 以上
- 外部工具（需另外安裝，**不透過 pip**）：
  - [FFmpeg](https://ffmpeg.org/download.html)：影音轉換與音訊提取
  - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)：板書文字辨識（Windows 請用 UB-Mannheim 版本，安裝時記得勾選中文語言包）

---

## 安裝

```bash
cd keytake-ai
pip install -r requirements.txt
```

複製環境變數範本並依需求填寫：

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

`.env` 預設設定（不需要 API Key 即可執行）：

```
SEMANTIC_SCORER_MODE=sbert
WHISPER_MODE=local
```

若要啟用 LLM 語意評分（效果更好，但需 API Key），將 `.env` 改為：

```
SEMANTIC_SCORER_MODE=llm
LLM_BACKEND=groq          # groq 有免費額度，推薦先試用
GROQ_API_KEY=gsk_xxx
```

---

## 快速開始

### 命令列模式

```bash
# 基本用法：輸出摘要至 output/ 目錄
python main.py "data/videos/lecture_01_math_homework.mp4"

# 搭配 Ground Truth 評估效能（Recall / FAR / TSR）
python main.py "data/videos/lecture_01_math_homework.mp4" \
    --gt data/annotations/gt_lecture_01_math_homework.json
```

### Web 介面模式

需要先啟動 Redis，再分別啟動 Celery worker 與 FastAPI：

```bash
# 1. 啟動 Redis（Windows 建議用 WSL 或 Docker）
#    docker run -d -p 6379:6379 redis

# 2. 啟動 Celery worker
celery -A src.platform.celery_app worker --loglevel=info

# 3. 啟動 FastAPI
uvicorn src.platform.api:app --reload
```

瀏覽器開啟 `http://localhost:8000`，上傳影片後等待處理完成，
即可下載摘要影片與精華片段索引。

---

## 目錄結構

```
keytake-ai/
├── main.py                     # 主流程入口（完整 pipeline）
├── config.py                   # 所有可調參數（α/β、門檻、路徑等）
├── requirements.txt
├── .env.example                # 環境變數範本
│
├── src/                        # 核心模組
│   ├── preprocessing/
│   │   └── preprocessor.py     # FFmpeg 轉換、音訊提取、SNR 降噪
│   │
│   ├── semantic/
│   │   ├── transcriber.py      # Whisper 語音轉錄（本地/遠端兩模式）
│   │   ├── scorer.py           # 語意評分（SBERT / LLM / TF-IDF）
│   │   ├── prosodic_analyzer.py# 韻律特徵（語速/音量/音高/停頓）
│   │   └── llm_cache.py        # LLM 回應快取（避免重複 API 呼叫）
│   │
│   ├── visual/
│   │   ├── hand_tracker.py     # MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── gesture_classifier.py # 雙層 GRU 手勢意圖分類
│   │   ├── visual_scorer.py    # OCR → SBERT 語意比對、IoU 備援
│   │   ├── clip_scorer.py      # CLIP 板書影像與口語關鍵字對齊
│   │   ├── visual_reliability.py # 四訊號加權估計視覺可靠度 R_visual
│   │   ├── text_detector.py    # pytesseract 文字區域偵測
│   │   ├── sbert_calculator.py # OCR 文字與 ASR 關鍵字語意相似度
│   │   └── srgan.py            # 模糊偵測 + SRGAN/bicubic 影像增強
│   │
│   ├── fusion/
│   │   ├── adaptive_fusion.py  # 自適應融合、語意感知滑動視窗
│   │   ├── evaluator.py        # Recall / Precision / F1 / FAR / TSR
│   │   ├── feature_extractor.py# MLP 融合用特徵擷取
│   │   └── mlp_fusion.py       # 可學習 MLP 融合模型（實驗性）
│   │
│   ├── output/
│   │   ├── video_exporter.py   # 剪輯輸出摘要影片、精華索引
│   │   └── study_materials.py  # 輸出 Markdown 筆記、心智圖、閃卡
│   │
│   └── platform/
│       ├── api.py              # FastAPI 後端（上傳/狀態/下載）
│       └── celery_app.py       # Celery 非同步任務
│
├── tools/                      # 標註工具與實驗腳本
│   ├── annotator.py            # 互動式標註工具，支援 Cohen's Kappa 計算
│   ├── export_transcript.py    # 從影片產生逐字稿 JSON
│   ├── run_ablation.py         # 單部影片消融實驗
│   ├── run_ablation_batch.py   # 多部影片批次消融（產生論文表格數據）
│   ├── run_single_eval.py      # 單部影片完整評估
│   ├── run_batch_eval.py       # 批次評估
│   ├── run_grid_search.py      # Grid Search 最佳 α/β
│   ├── run_loocv.py            # Leave-One-Out 交叉驗證
│   ├── run_analysis.py         # 參數敏感度分析（產生論文圖表）
│   ├── train_gesture_classifier.py # GRU 手勢分類器訓練
│   ├── train_mlp_fusion.py     # MLP 融合模型訓練
│   ├── make_figures.py         # 產生論文圖（長條圖、敏感度曲線、時間軸）
│   ├── collect_gesture_data.py # 收集手勢訓練資料
│   ├── check_env.py            # 環境檢查（確認 FFmpeg、Tesseract 等是否正確安裝）
│   └── diagnose.py             # 診斷單部影片的各模態分數分布
│
├── tests/                      # 單元測試
│   ├── smoke_test.py           # 端對端煙霧測試（不需 GPU）
│   ├── test_adaptive_fusion.py
│   ├── test_evaluator.py
│   ├── test_gesture_classifier.py
│   ├── test_sbert_calculator.py
│   ├── test_semantic_scorer.py
│   ├── test_text_detector.py
│   ├── test_visual_reliability.py
│   └── test_visual_scorer.py
│
├── data/
│   ├── videos/                 # 評估用教學影片（mp4，由 .gitignore 排除）
│   ├── annotations/            # Ground Truth 與逐字稿 JSON
│   │   ├── gt_*.json           # 人工標註 Ground Truth
│   │   ├── transcript_*.json   # Whisper 轉錄結果（由工具產生）
│   │   └── README.md           # 標註檔案格式說明
│   └── prompt_corpus/
│       └── corpus.py           # SBERT 語意評分用的教學提示語料庫
│
├── results/                    # 實驗結果
│   ├── ablation_batch.md       # 消融實驗彙整（論文表 2 數據來源）
│   ├── ablation_batch.json
│   ├── ablation_*.md/.json     # 各影片消融明細
│   ├── analysis_*.md/.json     # 參數敏感度分析結果
│   ├── features_*.json         # 各影片擷取的特徵向量
│   └── figures/                # 論文圖表（PNG）
│       ├── fig2_strategy_comparison.png
│       ├── fig3_beta_sensitivity.png
│       └── fig4_timeline.png
│
├── weights/                    # 訓練好的模型權重
│   ├── gesture_classifier.pth  # GRU 手勢分類器
│   └── mlp_fusion.pth          # MLP 融合模型（實驗性）
│
├── models/
│   └── hand_landmarker.task    # MediaPipe 手部偵測模型
│
└── docs/
    └── annotation_guideline.md # 標註準則（多人標註 Cohen's Kappa 用）
```

---

## 執行實驗

### 重現論文消融實驗（表 2）

```bash
# 批次跑四部影片，輸出彙整報告
python tools/run_ablation_batch.py
# 結果輸出至 results/ablation_batch.md
```

### 重現參數敏感度分析（圖 5）

```bash
python tools/run_analysis.py --video data/videos/test_10min.mp4 \
    --gt data/annotations/gt_test_10min.json
# 結果輸出至 results/analysis_test_10min.md
```

### 重新產生論文圖表

```bash
python tools/make_figures.py
# 輸出至 results/figures/
```

### 執行單元測試

```bash
python -m pytest tests/ -v
python tests/smoke_test.py    # 快速端對端測試，不需 GPU
```

---

## 標註工具使用

```bash
# 1. 從影片產生逐字稿
python tools/export_transcript.py \
    --video data/videos/lecture_01_math_homework.mp4 \
    --output data/annotations/

# 2. 互動式標註（依 Likert 1~5 分評估每段重要性）
python tools/annotator.py annotate \
    data/annotations/transcript_lecture_01_math_homework.json \
    --annotator A

# 3. 合併兩位標註者結果並計算 Cohen's Kappa
python tools/annotator.py merge \
    data/annotations/annotator_A_lecture_01.json \
    data/annotations/annotator_B_lecture_01.json \
    --output data/annotations/gt_lecture_01_math_homework.json
```

標註準則見 `docs/annotation_guideline.md`。

---

## 主要參數（config.py）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `WHISPER_MODEL` | `"base"` | tiny / base / small / medium / large |
| `SBERT_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 支援中文 |
| `ALPHA` | `0.6` | 語意權重（Grid Search 結果） |
| `BETA` | `0.4` | 視覺靜態權重 |
| `FUSION_SCORE_THRESHOLD` | `0.2` | 片段最低融合分數 |
| `MAX_SUMMARY_RATIO` | `0.6` | 摘要最長為原片的 60% |
| `GESTURE_INTENT_THRESHOLD` | `0.7` | 觸發視覺分析的最低意圖分數 |
| `CLIP_WEIGHT` | `0.3` | CLIP 在視覺分數中的混合比例 |
| `PROSODIC_WEIGHT` | `0.15` | 韻律特徵在融合中的影響權重 |

---

## 環境診斷

安裝完成後可執行環境檢查，確認所有外部工具與模型都正確設定：

```bash
python tools/check_env.py
```

輸出範例：

```
[✓] Python 3.11.x
[✓] FFmpeg 找到：C:\ffmpeg\bin\ffmpeg.exe
[✓] Tesseract 找到：C:\Program Files\Tesseract-OCR\tesseract.exe
[✓] PyTorch 2.x（CUDA 不可用，使用 CPU）
[✓] sentence-transformers 載入成功
[✓] MediaPipe 載入成功
[✓] hand_landmarker.task 存在
```
