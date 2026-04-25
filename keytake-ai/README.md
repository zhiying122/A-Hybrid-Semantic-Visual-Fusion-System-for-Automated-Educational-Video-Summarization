# KeyTake AI — 自動化教學精華擷取系統

多模態融合教學影片自動摘要系統，整合語意分析（Whisper + Sentence-BERT）與視覺事件偵測（MediaPipe 手部軌跡），透過適應性晚期融合生成語意連貫的摘要影片。

對應國科會計畫書第 4.2～4.4 節之系統實作。

---

## 系統需求

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html)（影片轉碼，需加入 PATH）
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)（Windows 安裝後路徑寫入 `config.py`）
- Redis（選填，Celery 非同步模式才需要）

---

## 安裝

```bash
cd keytake-ai
pip install -r requirements.txt
```

---

## 環境設定

複製並編輯 `.env`：

```bash
# .env 內容
WHISPER_MODE=local          # local（本地）或 remote（OpenAI API）
OPENAI_API_KEY=sk-...       # 遠端模式才需要填
REDIS_URL=redis://localhost:6379/0
```

Windows 路徑設定（`config.py` 第 8~13 行）：

```python
_ffmpeg_path = r"C:\ffmpeg-8.1-essentials_build\bin"
_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 快速開始

### 1. 對單一影片產生摘要

```bash
cd keytake-ai
python main.py path/to/lecture.mp4

# 附帶 Ground Truth 計算 Recall / FAR
tions/gt_example.json
```

### 2. 執行單元測試（不需要真實影片）

```bash
cd keytake-ai
python -m pytest tests/ -v
```

### 3. 啟動 Web 平台

```bash
# 本地 Whisper（預設）
cd keytake-ai
uvicorn src.platform.api:app --reload

# 遠端 Whisper API（快 10 倍）
set WHISPER_MODE=remote
set OPENAI_API_KEY=sk-你的key
uvicorn src.platform.api:app --reload

# 瀏覽器開啟
# hcalhost:8000
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

# Step 4：合併並計算 Cohen's Kappa（標註者間一致性）
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json
```

Ground Truth JSON 格式：

```json
{
  "video_id": "lecture_01",
  "ground_truth": [
    {"start_sec": 12.5, "end_sec": 28.0},
    {"start_sec": 145.0, "end_sec": 167.3}
  ]
}
```

---

## 實驗評估

```bash
# Grid Search 找最佳 α/β 權重
python tools/run_grid_search.py --annotations data/annotations/

# L泛化能力）
tations data/annotations/

# 批次評估（多部影片）
python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/

# 跨場域評估（黑板 / 投影片 / 高反光）
python tools/run_domain_eval.py --videos data/videos/ --annotations data/annotations/
```

---

## 提示語料庫說明

`data/prompt_corpus/corpus.py` 包含約 60 句教學引導語，涵蓋：

- 重點強調語句（「這邊非常重要」、「這是考試重點」）
- 因果推導引導（「導致這個結果的原因是」、「由此可知」）
- 板書視覺指引（「請看這張圖」、「我在黑板上寫的這個」）
- 總結複習語句（「總結一下」、「這一節的重點是」）
- 錯誤提醒（「這裡很多人會搞錯」）
- 英文教學場景（跨語言支援）
arm Rate | < 25% | 選中但完全未撞到 GT 的時間佔總時長比例 |
| 語意相似度 BERTScore | > 0.7 | 摘要文字與 GT 文字的向量相似度 |
| 時間節省率 Time Saving Rate | > 50% | (原始時長 - 摘要時長) / 原始時長 |

---

## 啟動 Celery（長影片非同步處理，選填）

```bash
# 需先啟動 Redis
# Windows 建議使用 Docker：
docker run -d -p 6379:6379 redis

# 啟動 Celery worker
celery -A src.platform.celery_app worker --loglevel=info
```

不啟動 Celery 時，系統自動降級為 threading 同步處理，功能不受影響。
執行腳本
│   ├── run_batch_eval.py          # 批次評估腳本
│   └── run_domain_eval.py         # 跨場域評估腳本（輸出 Markdown 報告）
└── tests/
    ├── smoke_test.py              # 不需真實影片的 pipeline 驗證
    ├── test_evaluator.py          # 評估指標 property-based tests
    ├── test_visual_scorer.py      # 視覺評分模組測試
    ├── test_text_detector.py      # 文字偵測模組測試
    └── test_sbert_calculator.py   # SBERT 計算器測試
```

---

## 評估目標（計畫書 4.4）

| 指標 | 目標 | 說明 |
|------|------|------|
| 重點召回率 Recall | > 70% | 系統選中片段與 GT 重疊 ≥ 50% 算命中 |
| 誤報率 False Aly          # 降級用任務狀態暫存
│       └── static/index.html      # 前端介面（三色時間軸 + 進度條）
├── data/
│   ├── prompt_corpus/corpus.py    # 教學提示語料庫（~60 句）
│   ├── videos/                    # 測試影片（blackboard / slides / challenging）
│   └── annotations/               # Ground Truth 標註（gt_*.json）
├── tools/
│   ├── annotator.py               # Ground Truth 標註工具（含 Cohen's Kappa）
│   ├── export_transcript.py       # 匯出 Whisper 逐字稿
│   ├── run_grid_search.py         # Grid Search 執行腳本
│   ├── run_loocv.py               # LOOCV 
│   │   ├── sbert_calculator.py    # 步驟三：OCR 與 ASR 語意相似度計算
│   │   └── srgan.py               # 步驟三（延伸）：SRGAN 超解析度增強
│   ├── fusion/
│   │   ├── adaptive_fusion.py     # 步驟四：晚期融合 + Grid Search + LOOCV
│   │   └── evaluator.py           # 評估指標：Recall / FAR / BERTScore / TSR
│   ├── output/
│   │   └── video_exporter.py      # FFmpeg 剪輯輸出 + 索引匯出（含 type 欄位）
│   └── platform/
│       ├── api.py                 # FastAPI 後端（Celery + threading 降級）
│       ├── celery_app.py          # Celery 非同步任務
│       ├── task_store.pSBERT + IoU 降級
│   │   ├── text_detector.py       # 步驟三：Tesseract 文字框偵測.py                      # 所有可調參數與閾值
├── requirements.txt               # Python 依賴套件
├── .env                           # 環境變數（不推上 GitHub）
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py        # 步驟一：FFmpeg 轉碼 + 動態降噪（SNR）
│   ├── semantic/
│   │   ├── transcriber.py         # 步驟二：Whisper 語音轉錄（本地/遠端雙模式）
│   │   └── scorer.py              # 步驟二：TF-IDF + SBERT 雙軌評分
│   ├── visual/
│   │   ├── hand_tracker.py        # 步驟三：MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── visual_scorer.py       # 步驟三：文字偵測優先 + OCR + ## 專案結構

```
keytake-ai/
├── main.py                        # 主流程 pipeline
├── config
這些語句參考自跨學科教學影片的常見引導模式，並依計畫書 4.2 步驟二的設計進行人工篩選與分類。

---

