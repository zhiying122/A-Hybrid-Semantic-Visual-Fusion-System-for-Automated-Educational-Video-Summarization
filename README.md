# KeyTake AI — 混合語意-視覺融合教學影片自動摘要系統

> A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization

將長篇教學錄影（如大學課堂板書講課）自動濃縮為精華片段，同時產出學習筆記、心智圖與閃卡。
系統整合語音語意分析、手部手勢意圖辨識、韻律特徵分析與 CLIP 跨模態對齊四種模態，
透過自適應置信度加權融合策略選取教學重點時刻。

---

## 系統架構總覽

```
輸入教學影片 (.mp4 / .avi / .mov)
  │
  ├── [步驟 1] 影音前處理
  │     FFmpeg 轉碼分離 → 音訊降噪（頻譜減法）
  │
  ├── [步驟 2] 語意分析
  │     Whisper 語音轉錄 → LLM 教學結構分析（6 種教學階段）→ S_text
  │     │                    支援 OpenAI / Groq / Ollama / SBERT / TF-IDF
  │     │
  │     └── LLM 快取層（避免重複呼叫，節省費用）
  │
  ├── [步驟 2.5] 韻律特徵提取
  │     librosa 分析語速、音量、音高變化、停頓比例 → S_prosodic
  │
  ├── [步驟 3] 視覺特徵提取
  │     MediaPipe 手部追蹤 → GRU 手勢意圖分類（5 類）
  │     │                     → ROI 裁切 → 模糊偵測 → SRGAN 增強
  │     │                     → 文字偵測 → OCR → SBERT 語意比對
  │     │                     → CLIP 圖文跨模態對齊（與 OCR 路徑混合）
  │     └── 視覺可靠度估測器 → S_visual + R_visual
  │
  ├── [步驟 4] 多模態融合
  │     自適應置信度加權晚期融合：
  │       β_eff = β × R_visual
  │       Score = (1 - β_eff) × S_text + β_eff × S_visual
  │       Score × (1 + 0.15 × (S_prosodic - 0.5) × 2)   ← 韻律加權
  │     或 MLP 可學習融合模型（監督式訓練，可選）
  │     → 語意感知滑動視窗選取最終片段
  │
  └── [步驟 5] 多元輸出
        摘要影片（FFmpeg 剪輯串接）
        精華片段索引 JSON（時間軸 + 教學階段標籤）
        Markdown 學習筆記
        心智圖 JSON
        Q&A 閃卡
```

---

## 環境需求

### 作業系統

Windows 10/11 或 Linux（macOS 未測試但理論可行）

### Python 版本

Python 3.11 以上（建議 3.12+）

### 必要外部工具

| 工具 | 用途 | 安裝方式 |
|------|------|---------|
| FFmpeg | 影音格式轉換、片段剪輯 | [官網下載](https://ffmpeg.org/download.html)，加入 PATH |
| Tesseract OCR | 板書文字辨識 | [Windows 安裝](https://github.com/UB-Mannheim/tesseract/wiki)，預設路徑 `C:\Program Files\Tesseract-OCR\` |

### 選用外部工具

| 工具 | 用途 | 說明 |
|------|------|------|
| Redis | Celery 任務佇列 | 僅 Web 平台非同步模式需要 |
| CUDA GPU | 加速推論 | PyTorch / Whisper / CLIP 可用 GPU，非必要 |

---

## 安裝步驟

### 1. 複製專案

```bash
git clone https://github.com/zhiying122/A-Hybrid-Semantic-Visual-Fusion-System-for-Automated-Educational-Video-Summarization.git
cd "A-Hybrid-Semantic-Visual-Fusion-System-for-Automated-Educational-Video-Summarization/keytake-ai"
```

### 2. 建立虛擬環境（建議）

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 3. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

主要依賴說明：

| 套件 | 版本 | 用途 |
|------|------|------|
| openai-whisper | 20250625 | 本地語音轉錄 |
| sentence-transformers | 5.3.0 | SBERT 語意相似度 |
| torch | 2.11.0 | 深度學習框架 |
| opencv-python | 4.13.0 | 影像處理 |
| mediapipe | 0.10.33 | 手部骨架追蹤 |
| pytesseract | 0.3.13 | OCR 文字辨識 |
| librosa | 0.10.2 | 韻律特徵分析 |
| transformers | 4.52.0 | CLIP 模型 |
| fastapi | 0.135.1 | Web API |
| bert-score | 0.3.13 | 評估指標 |

### 4. 設定環境變數

複製範例環境變數檔：

```bash
copy .env.example .env
```

編輯 `.env` 檔案：

```env
# === 語意評分模式（三選一）===
SEMANTIC_SCORER_MODE=sbert    # sbert / llm / tfidf

# === LLM 後端（SEMANTIC_SCORER_MODE=llm 時才需要）===
LLM_BACKEND=groq              # openai / groq / ollama

# OpenAI（付費，最準）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Groq（免費，速度快，推薦新手使用）
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# Ollama（本地部署，完全免費離線）
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# === Whisper 模式 ===
WHISPER_MODE=local            # local / remote（remote 需 OPENAI_API_KEY）
```

### 5. 驗證安裝

```bash
python -c "from config import ALPHA; print('安裝成功')"
```

---

## 使用方式

### 方式一：命令列直接執行

```bash
# 基本用法：處理一部影片
python main.py "你的教學影片.mp4"

# 帶 Ground Truth 評估
python main.py "lecture.mp4" --gt "data/annotations/gt_lecture.json"
```

輸出結果存放於 `output/` 目錄：
- `summary.mp4` — 摘要影片
- `index.json` — 精華片段索引（含時間軸、教學階段）
- `study_notes.md` — Markdown 學習筆記
- `mind_map.json` — 心智圖結構
- `flashcards.json` — Q&A 閃卡
- `chapters.json` — 章節標題

### 方式二：Web 平台

```bash
# 啟動 FastAPI 伺服器
uvicorn src.platform.api:app --reload --host 0.0.0.0 --port 8000
```

開啟瀏覽器訪問 `http://localhost:8000`，可透過網頁上傳影片並查看結果。

Web 平台功能：
- 拖曳上傳教學影片
- 即時顯示處理進度（步驟名稱 + 預估剩餘時間）
- 摘要影片播放器（含時間軸精華標記）
- 精華片段索引列表（點擊跳轉）
- 課程摘要卡片

### 方式三：Celery 非同步處理（適合長影片）

```bash
# 終端機 1：啟動 Redis（需事先安裝）
redis-server

# 終端機 2：啟動 Celery Worker
celery -A src.platform.celery_app worker --loglevel=info

# 終端機 3：啟動 FastAPI
uvicorn src.platform.api:app --reload
```

---

## 標註流程（建立 Ground Truth）

系統需要人工標註的 Ground Truth 進行效能評估與模型訓練。
採用雙盲專家評級 + Cohen's Kappa 一致性檢驗。

### 完整標註流程

```bash
# 步驟一：匯出逐字稿（Whisper 轉錄）
python tools/export_transcript.py --video data/videos/lecture_01.mp4 --output data/annotations/

# 步驟二：標註者 A 逐段評分（1~5 分）
python tools/annotator.py annotate data/annotations/transcript_lecture_01.json --annotator A

# 步驟三：標註者 B 逐段評分
python tools/annotator.py annotate data/annotations/transcript_lecture_01.json --annotator B

# 步驟四：合併標註、計算 Cohen's Kappa（標準 > 0.6）
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json --output data/annotations/gt_lecture_01.json
```

### 批次匯出逐字稿

```bash
python tools/export_transcript.py --videos data/videos/ --output data/annotations/
```

---

## 模型訓練與參數調校

### Grid Search 找最佳融合權重

```bash
python tools/run_grid_search.py --annotations data/annotations/ --output results/grid_search.json
```

找到最佳 α/β 後，更新 `config.py` 中的 `ALPHA` 和 `BETA`。

### Leave-One-Out 交叉驗證

```bash
python tools/run_loocv.py --annotations data/annotations/ --output results/loocv.json
```

確認模型在未見過的課程上的泛化穩定性。

### MLP 融合模型訓練（可選，v5 新增）

```bash
python tools/train_mlp_fusion.py --annotations data/annotations/ --output weights/mlp_fusion.pth
```

訓練完成後在 `config.py` 設定 `USE_MLP_FUSION = True` 啟用。

### GRU 手勢分類器訓練

```bash
python tools/train_gesture_classifier.py
```

---

## 評估與測試

### 批次評估（30 部影片）

```bash
python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/ --output results/batch_report.json
```

### 跨場域泛化測試

```bash
# 三種場域：blackboard（黑板）/ slides（投影片）/ challenging（高反光/低對比）
python tools/run_domain_eval.py --videos data/videos/ --annotations data/annotations/ --output results/domain_report.json
```

### 單元測試

```bash
pytest tests/ -v
```

### 煙霧測試（快速驗證核心功能）

```bash
python tests/smoke_test.py
```

---

## 效能評估指標目標

| 指標 | 目標 | 說明 |
|------|------|------|
| 重點召回率 (Recall) | > 70% | 標註重點被系統選中的比例 |
| 誤報率 (FAR) | < 25% | 非重點被誤選的時間佔比 |
| 語意相似度 (BERTScore) | > 0.70 | 摘要文字與原始重點的向量相似度 |
| 時間節省率 (TSR) | > 50% | 觀看摘要相比原始影片節省的時間 |

---

## 系統參數設定

所有參數集中於 `config.py`，以下為主要調校項目：

### 語意分析參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `WHISPER_MODEL` | Whisper 模型大小（tiny/base/small/medium/large） | `base` |
| `SBERT_MODEL` | SBERT 模型名稱 | `paraphrase-multilingual-MiniLM-L12-v2` |
| `TFIDF_TOP_K` | TF-IDF 取前 K 個關鍵詞 | `20` |

### 視覺分析參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `HAND_VARIANCE_THRESHOLD` | 手部滯留判斷閾值 | `50.0` |
| `ROI_SIZE` | 感興趣區域裁切大小 | `224` |
| `BLUR_THRESHOLD` | 模糊偵測閾值（觸發 SRGAN） | `100.0` |
| `CLIP_WEIGHT` | CLIP 分數混合權重 | `0.3` |

### 韻律分析參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `PROSODIC_WEIGHT` | 韻律分數在融合中的影響力 | `0.15` |
| `PROSODIC_SPEECH_RATE_WEIGHT` | 語速子權重 | `0.3` |
| `PROSODIC_VOLUME_WEIGHT` | 音量子權重 | `0.3` |
| `PROSODIC_PITCH_WEIGHT` | 音高子權重 | `0.25` |

### 融合參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `ALPHA` / `BETA` | 語意/視覺靜態權重（Grid Search 調校） | `0.5 / 0.5` |
| `FUSION_SCORE_THRESHOLD` | 片段保留最低融合分數 | `0.2` |
| `SLIDING_WINDOW_MIN_SEC` | 滑動視窗最小保留秒數 | `10` |
| `USE_MLP_FUSION` | 是否啟用 MLP 可學習融合 | `False` |

---

## 專案結構

```
keytake-ai/
├── main.py                           # 主流程 Pipeline（v5 四模態融合）
├── config.py                         # 所有可調參數集中管理
├── requirements.txt                  # Python 依賴清單
├── .env                              # 環境變數（API Key 等，不入版控）
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py           # 步驟 1：FFmpeg 轉碼 + SNR 動態降噪
│   │
│   ├── semantic/
│   │   ├── transcriber.py            # 步驟 2：Whisper 語音轉錄（本地/遠端）
│   │   ├── scorer.py                 # 步驟 2：三軌語意評分（LLM/SBERT/TF-IDF）
│   │   ├── prosodic_analyzer.py      # 步驟 2.5：韻律特徵分析（語速/音量/音高/停頓）
│   │   └── llm_cache.py             # LLM 評分結果快取（JSON 持久化）
│   │
│   ├── visual/
│   │   ├── hand_tracker.py           # 步驟 3：MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── gesture_classifier.py     # 步驟 3：GRU 手勢意圖分類器（5 類）
│   │   ├── visual_scorer.py          # 步驟 3：視覺分數計算（OCR + SBERT + SRGAN）
│   │   ├── clip_scorer.py            # 步驟 3：CLIP 圖文跨模態對齊
│   │   ├── text_detector.py          # 步驟 3：文字區域偵測
│   │   ├── sbert_calculator.py       # 步驟 3：OCR-ASR 語意相似度
│   │   ├── srgan.py                  # 步驟 3：超解析度增強（模糊板書）
│   │   └── visual_reliability.py     # 步驟 3：視覺可靠度估測器
│   │
│   ├── fusion/
│   │   ├── adaptive_fusion.py        # 步驟 4：自適應融合 + Grid Search + LOOCV
│   │   ├── mlp_fusion.py             # 步驟 4：MLP 可學習融合模型
│   │   └── evaluator.py             # 效能評估（Recall/FAR/BERTScore/TSR/F1）
│   │
│   ├── output/
│   │   ├── video_exporter.py         # 步驟 5：FFmpeg 摘要影片輸出
│   │   └── study_materials.py        # 步驟 5：學習素材產生器
│   │
│   └── platform/
│       ├── api.py                    # FastAPI Web 後端
│       ├── celery_app.py             # Celery 非同步任務佇列
│       ├── task_store.py             # 本地任務狀態儲存（開發用）
│       └── static/index.html         # 前端介面
│
├── tools/
│   ├── annotator.py                  # Ground Truth 雙盲標註工具
│   ├── export_transcript.py          # 批次逐字稿匯出
│   ├── run_grid_search.py            # Grid Search 最佳權重搜尋
│   ├── run_loocv.py                  # Leave-One-Out 交叉驗證
│   ├── run_batch_eval.py             # 批次效能評估
│   ├── run_domain_eval.py            # 跨場域泛化測試
│   ├── train_gesture_classifier.py   # GRU 手勢分類器訓練
│   └── train_mlp_fusion.py           # MLP 融合模型訓練
│
├── data/
│   ├── prompt_corpus/corpus.py       # 教學提示語料庫
│   ├── videos/                       # 測試影片存放處
│   ├── annotations/                  # 標註資料與 Ground Truth
│   └── gesture_dataset.json          # 手勢訓練資料集
│
├── weights/
│   ├── gesture_classifier.pth        # 訓練好的 GRU 手勢模型
│   └── mlp_fusion.pth               # 訓練好的 MLP 融合模型（可選）
│
├── tests/
│   ├── smoke_test.py                 # 煙霧測試（快速驗證）
│   ├── test_evaluator.py            # 評估模組測試
│   ├── test_visual_scorer.py        # 視覺分數測試
│   └── ...                          # 其他單元測試
│
└── results/                          # 評估報告輸出目錄
```

---

## 技術棧

| 類別 | 技術 | 說明 |
|------|------|------|
| 語音辨識 | OpenAI Whisper | 本地離線轉錄，支援遠端 API |
| 語意向量 | Sentence-BERT | 多語言語意相似度 |
| 大語言模型 | GPT-4o-mini / Llama3 / Groq | 教學結構分析 |
| 手部追蹤 | MediaPipe Hands | 即時骨架偵測 |
| 手勢分類 | GRU 雙層網路 | 5 類教學意圖辨識 |
| 跨模態對齊 | OpenAI CLIP | 圖文語意對齊 |
| 韻律分析 | librosa | 語速/音量/音高/停頓 |
| OCR | Tesseract | 繁中+英文板書辨識 |
| 超解析度 | Real-ESRGAN | 模糊板書增強 |
| 後端框架 | FastAPI + Celery | 非同步影片處理 |
| 評估指標 | BERTScore | 語意相似度評估 |

---

## 版本歷程

| 版本 | 主要變更 |
|------|---------|
| v1 | 基礎 pipeline：Whisper + TF-IDF + 手部追蹤 + 固定權重融合 |
| v2 | 視覺可靠度估測器、文字偵測優先策略、SBERT 語意計算、SRGAN 模糊增強 |
| v3 | GRU 手勢意圖分類器（5 類）、LLM 語意評分、多幀視覺取樣 |
| v4 | LLM 教學結構分析（6 種階段）、全課摘要、前端升級 |
| v5 | 韻律特徵分析、CLIP 跨模態對齊、MLP 可學習融合、LLM 快取、學習素材產生 |

---

## 常見問題

### Q: 沒有 GPU 可以跑嗎？

可以。所有模組都支援 CPU 執行，只是 Whisper 和 CLIP 在 CPU 上會比較慢。
建議 Whisper 使用 `base` 模型（`config.py` 中設定），處理一部 50 分鐘課程約需 5~10 分鐘。

### Q: 不想付 LLM API 費用怎麼辦？

三種免費方案：
1. 設定 `SEMANTIC_SCORER_MODE=sbert` — 使用本地 SBERT 模型（不需 API）
2. 設定 `LLM_BACKEND=groq` + 申請免費 Groq API key — 免費額度夠用
3. 設定 `LLM_BACKEND=ollama` + 本地跑 Ollama — 完全離線免費

### Q: librosa / transformers 沒裝會怎樣？

系統會自動降級：
- librosa 未裝 → 韻律分數固定為 0.5（不影響其他功能）
- transformers 未裝 → CLIP 功能跳過（只用 OCR→SBERT 路徑）
- sentence-transformers 未裝 → 自動降級為 TF-IDF 模式

### Q: 怎麼快速測試系統是否正常？

```bash
python tests/smoke_test.py
```

---

## 授權

本專案為學術研究用途。
