# KeyTake AI

多模態融合教學影片自動摘要系統，整合語音語意分析與視覺事件偵測，透過自適應置信度加權晚期融合產生語意連貫的精華影片。

---

## 系統概述

KeyTake AI 從一段完整的教學影片中，自動辨識並擷取關鍵教學片段，輸出包含：

- 精華摘要影片（FFmpeg 剪輯）
- 章節標題與時間索引
- Markdown 學習筆記
- 心智圖結構
- Q&A 閃卡

系統融合四種模態訊號：語意重要性（Whisper + LLM/SBERT）、視覺板書事件（MediaPipe + OCR）、韻律特徵（語速/音量/音高）、跨模態對齊（CLIP），以自適應權重決定各片段的保留與否。

---

## 系統架構

```
keytake-ai/
├── main.py                                 # 主流程 pipeline
├── config.py                               # 系統參數與閾值
├── requirements.txt                        # Python 依賴
├── .env.example                            # 環境變數範本
│
├── src/
│   ├── fusion/                             
│   │   ├── adaptive_fusion.py                  # 自適應置信度加權融合 + Grid Search + LOOCV
│   │   ├── evaluator.py                        # Recall / Precision / F1 / FAR / BERTScore / TSR
│   │   └── mlp_fusion.py                       # MLP 可學習融合模型（選用）
│   ├── platform/                           
│   │   ├── static/                             
│   │   │   └── index.html                          # 前端介面（三色時間軸 + 進度條）
│   │   ├── api.py                              # FastAPI 後端（含 Celery 降級為 threading）
│   │   ├── celery_app.py                       # Celery 非同步任務定義
│   │   └── task_store.py                       # 任務狀態暫存（無 Redis 時用）
│   ├── preprocessing/                      
│   │   └── preprocessor.py                     # FFmpeg 轉碼 + 動態 SNR 降噪
│   ├── semantic/                           
│   │   ├── llm_cache.py                        # LLM 呼叫快取（避免重複計費）
│   │   ├── prosodic_analyzer.py                # 韻律特徵提取（語速/音量/音高/停頓）
│   │   ├── scorer.py                           # 三軌語意評分（LLM / SBERT / TF-IDF）
│   │   └── transcriber.py                      # Whisper 語音轉錄（本地/遠端雙模式）
│   └── visual/                             
│   │   ├── clip_scorer.py                      # CLIP 視覺-文字跨模態對齊
│   │   ├── gesture_classifier.py               # GRU 手勢意圖分類器（5 類）
│   │   ├── hand_tracker.py                     # MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── sbert_calculator.py                 # OCR 與 ASR 語意相似度
│   │   ├── srgan.py                            # SRGAN 超解析度（預設 bicubic 降級）
│   │   ├── text_detector.py                    # Tesseract 文字框偵測
│   │   ├── visual_reliability.py               # 視覺可靠度估測器（四維度）
│   │   └── visual_scorer.py                    # OCR→SBERT 視覺分數 + IoU 降級
│
├── data/
│   ├── annotations/                        # Ground Truth 標註
│   │   └── README.md                           
│   ├── prompt_corpus/                      
│   │   └── corpus.py                           # 教學提示語料庫（約 60 句引導語）
│   ├── videos/                             # 測試影片（blackboard / slides / challenging）
│   │   └── README.md                           
│   ├── gesture_dataset.json                # 手勢訓練資料
│   ├── gesture_dataset_augmented.json      # 手勢增強資料集
│   └── gesture_dataset_real.json           # 真實場域手勢資料
│
├── tools/
│   ├── annotator.py                        # Ground Truth 標註（含 Cohen's Kappa）
│   ├── collect_gesture_data.py             # 手勢軌跡資料蒐集
│   ├── demo.py                             # 快速示範腳本
│   ├── diagnose.py                         # 環境診斷工具
│   ├── export_transcript.py                # 匯出 Whisper 逐字稿
│   ├── run_batch_eval.py                   # 批次評估（多部影片）
│   ├── run_demo_benchmark.py               # 模擬初步實驗結果
│   ├── run_domain_eval.py                  # 跨場域評估（輸出 Markdown 報告）
│   ├── run_gesture_pipeline.py             # 手勢辨識獨立測試
│   ├── run_grid_search.py                  # Grid Search 找最佳 α/β
│   ├── run_loocv.py                        # Leave-One-Out 交叉驗證
│   ├── train_gesture_classifier.py         # GRU 手勢分類器訓練
│   ├── train_mlp_fusion.py                 # MLP 融合模型訓練
│   └── update_readme.py                    # README 自動更新
│
├── tests/
│   ├── smoke_test.py                       # 不需真實影片的 pipeline 驗證
│   ├── test_adaptive_fusion.py             # 融合模組（property-based）
│   ├── test_evaluator.py                   # 評估指標（property-based）
│   ├── test_gesture_classifier.py          # 手勢分類器
│   ├── test_sbert_calculator.py            # SBERT 計算器
│   ├── test_semantic_scorer.py             # 語意評分模組
│   ├── test_text_detector.py               # 文字偵測模組
│   ├── test_visual_reliability.py          # 視覺可靠度估測器
│   └── test_visual_scorer.py               # 視覺評分模組
│
└── weights/
    └── gesture_classifier.pth              # 手勢分類器權重
```

---

## 環境需求

| 項目 | 說明 |
|------|------|
| Python | 3.10 以上 |
| FFmpeg | 影片轉碼，需加入系統 PATH |
| Tesseract OCR | Windows 安裝後路徑寫入 `config.py` |
| Redis | 選填，非同步 Celery 模式才需要 |
| GPU | 選填，SRGAN / CLIP / MLP 融合加速用 |

---

## 安裝步驟

```bash
cd keytake-ai
pip install -r requirements.txt
```

複製環境變數檔並依需求修改：

```bash
copy .env.example .env
```

`.env` 主要設定項：

```ini
# 語意評分模式：sbert（離線）/ llm（最佳效果）/ tfidf（最快）
SEMANTIC_SCORER_MODE=sbert

# LLM 後端：openai / groq（免費）/ ollama（離線）
LLM_BACKEND=groq

# Whisper：local（離線）/ remote（快 10 倍，需 API Key）
WHISPER_MODE=local
```

Windows 系統工具路徑（通常自動偵測，若失敗可設環境變數）：

```ini
# .env 中加入（僅在自動偵測失敗時才需要）
FFMPEG_PATH=C:\ffmpeg\bin
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 使用方式

### 單一影片摘要

```bash
python main.py path/to/lecture.mp4
```

附帶 Ground Truth 計算 Recall / FAR：

```bash
python main.py lecture.mp4 --gt data/annotations/example/gt_lecture01.json
```

### 啟動 Web 平台

```bash
uvicorn src.platform.api:app --reload
# 瀏覽器開啟 http://localhost:8000
```

遠端 Whisper 模式（速度快 10 倍）：

```bash
set WHISPER_MODE=remote
set OPENAI_API_KEY=sk-你的key
uvicorn src.platform.api:app --reload
```

### 執行測試

```bash
python -m pytest tests/ -v
```

---

## 標註流程（Ground Truth 建立）

系統採用雙盲專家評級建立黃金標準：

```bash
# 1. 轉錄影片產生逐字稿
python tools/export_transcript.py lecture.mp4 --output transcript.json

# 2. 兩位標註者分別以 Likert 1~5 分評分
python tools/annotator.py annotate transcript.json --annotator A
python tools/annotator.py annotate transcript.json --annotator B

# 3. 合併並計算 Cohen's Kappa（一致性 > 0.6 方可採用）
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json
```

---

## 實驗評估工具

```bash
# Grid Search 找最佳 α/β 融合權重
python tools/run_grid_search.py --annotations data/annotations/

# Leave-One-Out 交叉驗證（泛化能力）
python tools/run_loocv.py --annotations data/annotations/

# 批次評估（多部影片）
python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/

# 跨場域評估（黑板 / 投影片 / 高反光）
python tools/run_domain_eval.py --videos data/videos/ --annotations data/annotations/

# Demo Benchmark（模擬實驗結果，不需真實影片）
python tools/run_demo_benchmark.py
```

---

## 評估指標與設計目標

| 指標 | 目標 | 計算方式 |
|------|------|----------|
| 重點召回率 Recall | > 70% | 選中片段與 GT 時間重疊 ≥ 50% 算命中 |
| 誤報率 FAR | < 25% | 選中但未命中 GT 的時長佔總時長比例 |
| BERTScore | > 0.70 | 摘要文字與 GT 文字的向量語意相似度 |
| 時間節省率 TSR | > 50% | (原始時長 − 摘要時長) / 原始時長 |
| 精確度 Precision | — | 命中片段 / 所有選取片段 |
| F1-Score | — | Precision 與 Recall 之調和平均 |

---

## 核心演算法

### 自適應置信度加權晚期融合

傳統靜態加權：`Score = α × S_text + β × S_visual`

本系統改為動態調整：

```
β_eff = β × R_visual
α_eff = 1 − β_eff
Score = α_eff × S_text + β_eff × S_visual + prosodic_boost
```

其中 `R_visual` 由視覺可靠度估測器從手部穩定性、偵測信心度、板書密度、手板接近度四個維度計算，當視覺品質低時自動降權視覺、提升語意，反之亦然。

### 多幀視覺取樣

每個語音片段在 25% / 50% / 75% 三個時間點各取一幀，取最高視覺分數作為該片段代表，降低單幀偶然性影響。

### 語意感知滑動視窗

避免因門檻切割造成語句中斷，以最小保留秒數為約束合併連續片段，確保輸出影片語意完整。

### MLP 可學習融合（進階，選用）

除了解析式公式，系統亦支援以 MLP 監督式學習自動找到最佳特徵組合權重。此功能需要先以 Ground Truth 標註資料訓練模型：

```bash
# 訓練 MLP 融合模型（需要已標註的影片資料）
python tools/train_mlp_fusion.py --annotations data/annotations/

# 訓練完成後在 config.py 中啟用
# USE_MLP_FUSION = True
```

未訓練時系統自動使用解析式融合公式，不影響正常運作。

---

## 啟動 Celery（選填，長影片非同步處理）

```bash
# 啟動 Redis
docker run -d -p 6379:6379 redis

# 啟動 Worker
celery -A src.platform.celery_app worker --loglevel=info
```

不啟動時系統自動降級為 threading 同步處理，功能不受影響。

---

## 提示語料庫

`data/prompt_corpus/corpus.py` 包含約 60 句教學引導語，涵蓋：

- 重點強調（「這邊非常重要」、「考試一定會考」）
- 因果推導（「由此可知」、「導致這個結果的原因是」）
- 板書指引（「請看這張圖」、「黑板上這個公式」）
- 總結複習（「總結一下」、「這節重點」）
- 錯誤提醒（「這裡很多人會搞錯」）
- 英文場景（跨語言支援）

這些語句用於 SBERT 模式下計算片段與教學引導語的餘弦相似度，作為語意重要性的評估基準。

---

## 技術棧

| 領域 | 技術 |
|------|------|
| 語音轉錄 | OpenAI Whisper（local / remote） |
| 語意評分 | Sentence-BERT / GPT-4o-mini / Groq Llama3 / Ollama |
| 手部追蹤 | MediaPipe Hands + 卡爾曼濾波 |
| 文字辨識 | Tesseract OCR |
| 跨模態對齊 | OpenAI CLIP (ViT-B/32) |
| 韻律分析 | librosa（語速/音量/音高/停頓） |
| 超解析度 | SRGAN（ROI-focused，預設 bicubic 降級） |
| 融合策略 | 自適應加權 / MLP 可學習融合 |
| 影片處理 | FFmpeg + OpenCV |
| Web 平台 | FastAPI + Celery + Redis |
| 測試框架 | pytest + Hypothesis（property-based testing） |
| 中文處理 | jieba 斷詞 |

---

## 授權

本專案為研究用途開發。
