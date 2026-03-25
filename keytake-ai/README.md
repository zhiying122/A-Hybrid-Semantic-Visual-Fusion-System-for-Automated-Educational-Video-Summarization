# KeyTake AI — 自動化教學精華擷取系統

多模態融合教學影片自動摘要系統，整合語意分析（Whisper + Sentence-BERT）與視覺事件偵測（MediaPipe 手部軌跡），透過適應性晚期融合生成語意連貫的摘要影片。

---

## 安裝

```bash
cd keytake-ai
pip install -r requirements.txt
```

> FFmpeg 需另外安裝：https://ffmpeg.org/download.html  
> Tesseract OCR 需另外安裝：https://github.com/UB-Mannheim/tesseract/wiki（Windows）

---

## 快速開始

### 1. 對單一影片產生摘要

```bash
cd keytake-ai
python main.py path/to/lecture.mp4
```

### 2. 執行 Smoke Test（不需要真實影片）

```bash
cd keytake-ai
python tests/smoke_test.py
```

### 3. 啟動 Web 平台

```bash
# 啟動 API server
cd keytake-ai
uvicorn src.platform.api:app --reload

# 瀏覽器開啟
# http://localhost:8000
```

---

## 標註流程（Ground Truth 建立）

```bash
# Step 1：轉錄影片產生逐字稿
python tools/export_transcript.py lecture.mp4 --output transcript.json

# Step 2：標註者 A 評分
python tools/annotator.py annotate transcript.json --annotator A

# Step 3：標註者 B 評分
python tools/annotator.py annotate transcript.json --annotator B

# Step 4：合併並計算 Cohen's Kappa
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json
```

---

## 實驗評估

```bash
# Grid Search 找最佳 α/β 權重
python tools/run_grid_search.py --annotations data/annotations/

# Leave-One-Out 交叉驗證
python tools/run_loocv.py --annotations data/annotations/

# 批次評估 30 部影片
python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/
```

---

## 專案結構

```
keytake-ai/
├── main.py                        # 主流程 pipeline
├── config.py                      # 所有可調參數
├── requirements.txt
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py        # 步驟一：FFmpeg 轉碼 + 動態降噪
│   ├── semantic/
│   │   ├── transcriber.py         # 步驟二：Whisper 語音轉錄
│   │   └── scorer.py              # 步驟二：TF-IDF + SBERT 雙軌評分
│   ├── visual/
│   │   ├── hand_tracker.py        # 步驟三：MediaPipe 手部追蹤
│   │   ├── visual_scorer.py       # 步驟三：OCR + IoU 降級備援
│   │   └── srgan.py               # 步驟三（延伸）：超解析度增強
│   ├── fusion/
│   │   ├── adaptive_fusion.py     # 步驟四：晚期融合 + Grid Search + LOOCV
│   │   └── evaluator.py           # 評估指標：Recall / BERTScore / TSR
│   ├── output/
│   │   └── video_exporter.py      # FFmpeg 剪輯輸出 + 索引匯出
│   └── platform/
│       ├── api.py                 # FastAPI 後端
│       ├── celery_app.py          # Celery 非同步任務
│       ├── task_store.py          # 降級用任務狀態暫存
│       └── static/index.html      # 前端介面
├── data/
│   └── prompt_corpus/corpus.py    # 教學提示語料庫（~60 句）
├── tools/
│   ├── annotator.py               # Ground Truth 標註工具（含 Cohen's Kappa）
│   ├── export_transcript.py       # 匯出 Whisper 逐字稿
│   ├── run_grid_search.py         # Grid Search 執行腳本
│   ├── run_loocv.py               # LOOCV 執行腳本
│   └── run_batch_eval.py          # 批次評估腳本
└── tests/
    └── smoke_test.py              # 不需真實影片的 pipeline 驗證
```

---

## 評估目標（計畫書 4.4）

| 指標 | 目標 |
|------|------|
| 重點召回率 Recall | > 70% |
| 誤報率 False Alarm Rate | < 25% |
| 語意相似度 BERTScore | > 0.7 |
| 時間節省率 Time Saving Rate | > 50% |

---

## 啟動 Celery（長影片非同步處理）

```bash
# 需先安裝並啟動 Redis
# Windows 建議使用 WSL 或 Docker：
# docker run -d -p 6379:6379 redis

celery -A src.platform.celery_app worker --loglevel=info
```
