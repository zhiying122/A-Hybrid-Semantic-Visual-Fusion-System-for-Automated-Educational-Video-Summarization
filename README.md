# KeyTake AI — 混合語意-視覺融合教學影片自動摘要系統

> **A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization**

將長篇教學錄影（大學課堂板書、投影片講解等）自動濃縮為精華片段，同時產出結構化學習素材
（章節標題、Markdown 筆記、心智圖、Q&A 閃卡）。系統整合**四種模態**——語音語意、手部手勢意圖、
語音韻律、CLIP 圖文跨模態對齊——透過**自適應置信度加權融合**，選取教學重點時刻。

本文件為專案主文件，涵蓋：專案理念 → 系統架構 → 安裝 → 使用（CLI / Web / 非同步）→
標註流程 → 模型訓練 → 評估測試 → 參數設定 → 專案結構 → 驗證結果 → 常見問題。

---

## 目錄

- [這個專案在做什麼](#這個專案在做什麼)
- [核心特色](#核心特色)
- [系統架構總覽](#系統架構總覽)
- [四模態融合原理](#四模態融合原理)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [使用方式](#使用方式)
  - [方式一：命令列](#方式一命令列直接執行)
  - [方式二：Web 平台](#方式二web-平台)
  - [方式三：Celery 非同步](#方式三celery-非同步處理適合長影片)
- [輸出檔案說明](#輸出檔案說明)
- [Web API 參考](#web-api-參考)
- [下載測試影片](#下載測試影片)
- [標註流程（建立-ground-truth）](#標註流程建立-ground-truth)
- [模型訓練與參數調校](#模型訓練與參數調校)
- [評估與測試](#評估與測試)
- [效能評估指標目標](#效能評估指標目標)
- [系統參數設定](#系統參數設定)
- [專案結構](#專案結構)
- [技術棧](#技術棧)
- [驗證結果](#驗證結果)
- [版本歷程](#版本歷程)
- [常見問題](#常見問題)

---

## 這個專案在做什麼

一堂大學課程往往長達 50 分鐘到數小時，但真正的「重點時刻」——老師強調的定義、關鍵推導、
板書書寫、經典範例——只佔其中一小部分。學生複習時得反覆快轉找重點，效率低落。

**KeyTake AI 自動找出這些重點時刻，並把它變成三種東西：**

1. **一部濃縮的精華影片**（保證比原片短，預設節省 ≥ 40% 時間）
2. **一份索引 JSON**（每段重點的時間軸 + 教學階段標籤，可點擊跳轉）
3. **一整組學習教材**（章節標題、Markdown 筆記、心智圖、Q&A 閃卡）

與只做語音摘要的工具不同，KeyTake AI 同時看「老師說了什麼」和「老師在黑板上做了什麼」，
用**手勢意圖**和**板書內容**補強純語音判斷的不足，因此在板書課堂特別有效。

---

## 核心特色

| 特色 | 說明 |
|------|------|
| **四模態融合** | 語意 + 視覺（手勢/板書）+ 韻律 + CLIP 圖文對齊，互相補強 |
| **自適應置信度加權** | 視覺不可靠（手部離開/無板書）時自動降低視覺權重，避免誤判 |
| **手勢意圖辨識** | GRU 分類器辨識 5 種教學手勢（書寫/強調/指向/過渡/無意義） |
| **多後端語意評分** | LLM（OpenAI/Groq/Ollama）/ SBERT / TF-IDF 三軌，可離線可付費 |
| **保證濃縮** | 任何長度影片輸出都保證比原片短（`MAX_SUMMARY_RATIO`） |
| **優雅降級** | librosa/CLIP/SBERT 缺任一都會自動降級，不會整個掛掉 |
| **LLM 快取** | 相同片段不重複呼叫 API，穩定又省錢 |
| **學習素材產生** | 自動把重點片段轉成筆記、心智圖、閃卡 |
| **Web 平台** | FastAPI + Celery 非同步處理，拖曳上傳、即時進度 |

---

## 系統架構總覽

```
輸入教學影片 (.mp4 / .avi / .mov / .webm)
  │
  ├── [步驟 1] 影音前處理  (src/preprocessing/preprocessor.py)
  │     FFmpeg 轉碼分離音視訊 → 依 SNR 決定是否頻譜減法降噪
  │
  ├── [步驟 2] 語意分析  (src/semantic/)
  │     Whisper 語音轉錄（本地/遠端）
  │       → LLM 教學結構分析（6 種教學階段）+ 重要性評分 → S_text
  │       → 全課脈絡分析（課程標題與摘要）
  │     後端可選：OpenAI / Groq / Ollama / SBERT / TF-IDF
  │     LLM 快取層（llm_cache.py）避免重複呼叫
  │
  ├── [步驟 2.5] 韻律特徵提取  (src/semantic/prosodic_analyzer.py)
  │     librosa 分析語速、音量、音高變化、停頓比例 → S_prosodic
  │
  ├── [步驟 3] 視覺特徵提取  (src/visual/)  ── 每段取頭/中/尾三幀
  │     MediaPipe 手部追蹤 + 卡爾曼濾波
  │       → GRU 手勢意圖分類（5 類）
  │       → ROI 裁切 → Laplacian 模糊偵測 →（模糊時）SRGAN 超解析度增強
  │       → 文字區域偵測 → OCR → SBERT 比對 ASR 關鍵字
  │       → CLIP 圖文跨模態對齊（與 OCR→SBERT 路徑混合）
  │     視覺可靠度估測器 → S_visual + R_visual
  │
  ├── [步驟 4] 多模態融合  (src/fusion/)
  │     自適應置信度加權晚期融合：
  │       β_eff = β × R_visual
  │       base  = (1 - β_eff) × S_text + β_eff × S_visual
  │       score = base × (1 + 0.15 × (S_prosodic - 0.5) × 2)   ← 韻律加權
  │     或 MLP 可學習融合模型（監督式訓練，可選）
  │       → 語意感知滑動視窗選段 → 保證濃縮上限修剪
  │
  └── [步驟 5] 多元輸出  (src/output/)
        summary 影片（FFmpeg 剪輯串接）
        index.json（時間軸 + 教學階段標籤）
        study_notes.md（Markdown 筆記）
        mind_map.json（心智圖）
        flashcards.json（Q&A 閃卡）
        chapters.json（章節標題）
```

---

## 四模態融合原理

系統對每個時間片段計算四種分數，最後融合成單一「重要性分數」：

| 模態 | 分數 | 來源 | 直覺 |
|------|------|------|------|
| 語意 | `S_text` | Whisper 轉錄 → LLM/SBERT 評分 | 老師說的內容有多重要 |
| 視覺 | `S_visual` | 手勢意圖 × 板書 OCR/CLIP 對齊 | 老師在黑板上做的動作有多關鍵 |
| 韻律 | `S_prosodic` | librosa 語速/音量/音高/停頓 | 老師講話的方式（強調時通常放慢加重） |
| 可靠度 | `R_visual` | 手部穩定度、板書密度、手板距離 | 這一刻「看畫面」值不值得信任 |

**自適應加權的核心思想**：當視覺線索不可靠（手離開畫面、沒有板書），
系統會自動把權重壓回語意；當老師正在板書、手勢穩定，視覺權重才會拉高。
這比固定權重更能應對真實課堂的多變場景。

融合公式（解析式，預設）：

```
β_eff = BETA × R_visual
base  = (1 − β_eff) × S_text + β_eff × S_visual
score = clip( base × (1 + PROSODIC_WEIGHT × (S_prosodic − 0.5) × 2),  0, 1 )
```

進階可改用 **MLP 可學習融合**（`USE_MLP_FUSION=True`），以標註資料監督式學習取代固定公式。

---

## 環境需求

### 作業系統
Windows 10/11 或 Linux（macOS 理論可行，未完整測試）。

### Python 版本
Python 3.11 以上（本專案於 **Python 3.14** 驗證通過）。

### 必要外部工具

| 工具 | 用途 | 安裝方式 |
|------|------|---------|
| **FFmpeg** | 影音轉碼、片段剪輯 | [官網下載](https://ffmpeg.org/download.html)，加入系統 PATH |
| **Tesseract OCR** | 板書文字辨識 | [Windows 安裝包](https://github.com/UB-Mannheim/tesseract/wiki)，預設路徑 `C:\Program Files\Tesseract-OCR\` |

> `config.py` 會自動偵測 FFmpeg / Tesseract 路徑（環境變數 → PATH → 常見安裝位置）。
> 若裝在非標準位置，可用 `.env` 的 `FFMPEG_PATH` / `TESSERACT_PATH` 指定。

### 選用外部工具

| 工具 | 用途 | 說明 |
|------|------|------|
| Redis | Celery 任務佇列 | 僅 Web 平台「非同步模式」需要 |
| CUDA GPU | 加速推論 | Whisper / SBERT / CLIP 可用 GPU，非必要（CPU 也能跑） |
| Ollama | 本地 LLM | 想完全離線做 LLM 語意評分時安裝 |

---

## 安裝步驟

### 1. 取得專案

```bash
git clone <本專案網址>
cd "A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization/keytake-ai"
```

### 2. 建立虛擬環境（建議）

```powershell
python -m venv venv
venv\Scripts\activate          # Windows PowerShell / CMD
# source venv/bin/activate     # Linux / macOS
```

### 3. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

主要依賴：

| 套件 | 版本 | 用途 |
|------|------|------|
| openai-whisper | 20250625 | 本地語音轉錄 |
| sentence-transformers | 5.3.0 | SBERT 語意相似度 |
| torch / torchvision | 2.11.0 / 0.26.0 | 深度學習框架 |
| opencv-python | 4.13.0 | 影像處理、卡爾曼濾波 |
| mediapipe | 0.10.33 | 手部骨架追蹤 |
| pytesseract | 0.3.13 | OCR（需 Tesseract 本體） |
| librosa | 0.10.2 | 韻律特徵分析 |
| transformers | 4.52+ | CLIP 圖文對齊 |
| bert-score | 0.3.13 | BERTScore 評估指標 |
| fastapi / uvicorn | 0.135.1 / 0.42.0 | Web 後端 |
| celery / redis | 5.6.2 / 7.4.0 | 非同步任務佇列 |
| groq | 0.30.0 | Groq LLM 後端（免費，選填） |
| jieba / scikit-learn | 0.42.1 / 1.8.0 | 中文斷詞、TF-IDF |

> **選填套件**：`librosa`、`groq`、`transformers` 未安裝時系統會自動降級
> （韻律固定 0.5 / 無 Groq 後端 / 跳過 CLIP），但建議全裝以取得完整效果。

### 4. 設定環境變數

```powershell
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS
```

編輯 `.env`：

```env
# === 語意評分模式（三選一）===
SEMANTIC_SCORER_MODE=sbert     # sbert（本地，推薦）/ llm（最準）/ tfidf（最快）

# === LLM 後端（SEMANTIC_SCORER_MODE=llm 時才需要）===
LLM_BACKEND=groq               # openai / groq / ollama
# OPENAI_API_KEY=sk-xxxx       # OpenAI（付費，最準）
# GROQ_API_KEY=gsk_xxxx        # Groq（免費額度，推薦試用）
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=llama3

# === Whisper 語音轉錄模式 ===
WHISPER_MODE=local             # local（離線免費）/ remote（需 OPENAI_API_KEY，快 10 倍）

# === 系統工具路徑（選填，通常自動偵測）===
# FFMPEG_PATH=C:\ffmpeg\bin
# TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 5. 驗證安裝

```bash
python -c "from config import ALPHA; print('設定載入成功，ALPHA =', ALPHA)"
python tests/smoke_test.py
```

看到 `所有測試通過 ✓` 即代表核心模組串接正常。

---

## 快速開始

```bash
# 進入專案子目錄
cd keytake-ai

# 處理一部教學影片
python main.py "你的教學影片.mp4"
```

處理完成後，`output/` 目錄會出現摘要影片與學習素材。詳見[輸出檔案說明](#輸出檔案說明)。

---

## 使用方式

### 方式一：命令列直接執行

```bash
# 基本用法
python main.py "lecture.mp4"

# 帶 Ground Truth 做效能評估（計算 Recall / FAR）
python main.py "lecture.mp4" --gt "data/annotations/gt_lecture.json"
```

`--gt` 支援兩種格式：
- annotator.py 產出的 `{"ground_truth": [{"start": ..., "end": ...}]}`
- 簡單陣列 `[[10.5, 20.0], [45.0, 60.0]]`

### 方式二：Web 平台

```bash
uvicorn src.platform.api:app --reload --host 0.0.0.0 --port 8000
```

瀏覽器開啟 `http://localhost:8000`：
- 拖曳上傳教學影片
- 即時處理進度（步驟名稱）
- 摘要影片播放器 + 精華片段索引（點擊跳轉）
- 課程摘要卡片、學習素材下載

### 方式三：Celery 非同步處理（適合長影片）

需要三個終端機：

```bash
# 終端機 1：啟動 Redis（需先安裝）
redis-server

# 終端機 2：啟動 Celery Worker
celery -A src.platform.celery_app worker --loglevel=info

# 終端機 3：啟動 FastAPI
uvicorn src.platform.api:app --reload
```

上傳後拿到 `task_id`，前端會自動輪詢 `/status/{task_id}` 直到完成。

---

## 輸出檔案說明

命令列模式輸出於 `output/`；Web 模式輸出於 `tmp/outputs/{task_id}/`。

| 檔案 | 內容 |
|------|------|
| `video.mp4` / `summary.mp4` | 摘要精華影片（FFmpeg 剪輯串接） |
| `audio.wav` | 前處理後的音訊（供轉錄用） |
| `index.json` | 精華片段索引（時間軸 + 教學階段 + LLM 摘要句 + 課程摘要） |
| `chapters.json` | 章節標題 |
| `study_notes.md` | Markdown 學習筆記 |
| `mind_map.json` | 心智圖結構 |
| `flashcards.json` | Q&A 閃卡 |
| `eval_report.json` | （帶 `--gt` 時）Recall / FAR / TSR 評估報告 |

---

## Web API 參考

FastAPI 後端（`src/platform/api.py`）提供的端點：

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/` | 前端網頁介面 |
| `POST` | `/upload` | 上傳影片，回傳 `task_id`（交由 Celery 非同步處理） |
| `GET` | `/status/{task_id}` | 查詢任務處理狀態與進度 |
| `GET` | `/download/{task_id}` | 下載摘要影片 |
| `GET` | `/materials/{task_id}/{filename}` | 下載指定學習素材（notes/mind_map/flashcards 等） |

啟動後可於 `http://localhost:8000/docs` 查看 Swagger 互動式文件。

---

## 下載測試影片

專案內建兩支下載工具，方便快速取得驗證用的公開教學影片。

### 從 Internet Archive 下載（推薦，免 JS runtime）

```bash
# 自動搜尋 lecture 類影片並下載 3 部，各裁切前 90 秒
python tools/fetch_archive_videos.py --count 3 --clip-seconds 90

# 指定 archive.org identifier
python tools/fetch_archive_videos.py --identifiers <id1> <id2> --clip-seconds 120

# 完整下載（不裁切）
python tools/fetch_archive_videos.py --count 2 --clip-seconds 0
```

### 從 YouTube 下載（需 yt-dlp + JS runtime）

```bash
python tools/fetch_test_videos.py --urls "https://www.youtube.com/watch?v=XXXX" --clip-seconds 90
```

> YouTube 近期要求 JS runtime（deno）且對機房 IP 有速率限制，可能出現 403。
> 若失敗，改用 `fetch_archive_videos.py` 或直接使用 `data/videos/test_10min.mp4`。

下載結果存於 `data/videos/downloaded/`，可直接餵給 `main.py`：

```bash
python main.py "data/videos/downloaded/<檔名>.mp4"
```

---

## 標註流程（建立 Ground Truth）

效能評估與模型訓練需要人工標註的 Ground Truth，採**雙盲專家評級 + Cohen's Kappa 一致性檢驗**。

```bash
# 步驟一：匯出逐字稿（Whisper 轉錄）
python tools/export_transcript.py --video data/videos/lecture_01.mp4 --output data/annotations/

# 批次匯出整個資料夾
python tools/export_transcript.py --videos data/videos/ --output data/annotations/

# 步驟二/三：兩位標註者各自逐段評分（1~5 分）
python tools/annotator.py annotate data/annotations/transcript_lecture_01.json --annotator A
python tools/annotator.py annotate data/annotations/transcript_lecture_01.json --annotator B

# 步驟四：合併標註、計算 Cohen's Kappa（標準 > 0.6 視為一致）
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json \
    --output data/annotations/gt_lecture_01.json
```

---

## 模型訓練與參數調校

```bash
# Grid Search 搜尋最佳融合權重 α/β，找到後更新 config.py
python tools/run_grid_search.py --annotations data/annotations/ --output results/grid_search.json

# Leave-One-Out 交叉驗證（確認泛化穩定性）
python tools/run_loocv.py --annotations data/annotations/ --output results/loocv.json

# 訓練 GRU 手勢意圖分類器 → weights/gesture_classifier.pth
python tools/train_gesture_classifier.py

# 訓練 MLP 融合模型 → weights/mlp_fusion.pth（訓練後在 config.py 設 USE_MLP_FUSION=True）
python tools/train_mlp_fusion.py --annotations data/annotations/ --output weights/mlp_fusion.pth
```

---

## 評估與測試

```bash
# 批次效能評估（多部影片）
python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/ \
    --output results/batch_report.json

# 跨場域泛化測試：blackboard（黑板）/ slides（投影片）/ challenging（高反光低對比）
python tools/run_domain_eval.py --videos data/videos/ --annotations data/annotations/ \
    --output results/domain_report.json

# 單元測試（完整）
python -m pytest tests/ -v

# 只跑快速測試（不含載入大模型的 property-based 測試）
python -m pytest tests/test_adaptive_fusion.py tests/test_evaluator.py \
    tests/test_visual_reliability.py tests/test_gesture_classifier.py \
    tests/test_semantic_scorer.py tests/test_text_detector.py -v

# 煙霧測試（最快，端到端邏輯驗證，不需真實影片）
python tests/smoke_test.py
```

> `tests/test_sbert_calculator.py` 與 `tests/test_visual_scorer.py` 為 property-based
> （hypothesis）測試，會實際載入 SBERT/CLIP 模型並跑數百組例子，執行時間較長屬正常現象。

---

## 效能評估指標目標

| 指標 | 目標 | 說明 |
|------|------|------|
| 重點召回率 (Recall) | > 70% | 標註重點被系統選中的比例 |
| 誤報率 (FAR) | < 25% | 非重點被誤選的時間佔比 |
| 語意相似度 (BERTScore) | > 0.70 | 摘要文字與原始重點的向量相似度 |
| 時間節省率 (TSR) | > 50% | 觀看摘要相比原片節省的時間 |

---

## 系統參數設定

所有參數集中於 `config.py`，主要調校項目：

### 語意分析

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `WHISPER_MODEL` | Whisper 模型大小（tiny/base/small/medium/large） | `base` |
| `SBERT_MODEL` | SBERT 模型名稱 | `paraphrase-multilingual-MiniLM-L12-v2` |
| `TFIDF_TOP_K` | TF-IDF 取前 K 個關鍵詞 | `20` |
| `SBERT_SIMILARITY_THRESHOLD` | 與提示語料庫最低餘弦相似度 | `0.6` |

### 視覺分析

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `HAND_VARIANCE_THRESHOLD` | 手部滯留判斷閾值 | `50.0` |
| `ROI_SIZE` | 感興趣區域裁切大小 | `224` |
| `BLUR_THRESHOLD` | 模糊偵測閾值（觸發 SRGAN） | `100.0` |
| `SRGAN_SCALE` | 超解析度放大倍數 | `4` |
| `GESTURE_INTENT_THRESHOLD` | 觸發視覺分析的最低意圖分數 | `0.7` |
| `CLIP_WEIGHT` | CLIP 分數混合權重 | `0.3` |

### 韻律分析

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `PROSODIC_WEIGHT` | 韻律分數在融合中的影響力 | `0.15` |
| `PROSODIC_SPEECH_RATE_WEIGHT` | 語速子權重 | `0.3` |
| `PROSODIC_VOLUME_WEIGHT` | 音量子權重 | `0.3` |
| `PROSODIC_PITCH_WEIGHT` | 音高子權重 | `0.25` |
| `PROSODIC_PAUSE_WEIGHT` | 停頓子權重 | `0.15` |

### 融合與濃縮

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `ALPHA` / `BETA` | 語意 / 視覺靜態權重（α + β = 1，Grid Search 調校） | `0.6 / 0.4` |
| `FUSION_SCORE_THRESHOLD` | 片段保留最低融合分數 | `0.2` |
| `SLIDING_WINDOW_MIN_SEC` | 滑動視窗最小保留秒數 | `10` |
| `MAX_SUMMARY_RATIO` | 摘要時長上限比例（保證濃縮） | `0.6` |
| `MIN_SUMMARY_SEC` | 摘要最短輸出秒數 | `2.0` |
| `USE_MLP_FUSION` | 是否啟用 MLP 可學習融合 | `False` |

---

## 專案結構

```
keytake-ai/
├── main.py                           # 主流程 Pipeline（v5 四模態融合）
├── config.py                         # 所有可調參數集中管理 + 工具路徑自動偵測
├── requirements.txt                  # Python 依賴清單
├── .env / .env.example               # 環境變數（API Key 等，.env 不入版控）
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py           # 步驟 1：FFmpeg 轉碼 + SNR 動態降噪
│   ├── semantic/
│   │   ├── transcriber.py            # 步驟 2：Whisper 語音轉錄（本地/遠端）
│   │   ├── scorer.py                 # 步驟 2：三軌語意評分（LLM/SBERT/TF-IDF）
│   │   ├── prosodic_analyzer.py      # 步驟 2.5：韻律特徵（語速/音量/音高/停頓）
│   │   └── llm_cache.py              # LLM 評分結果快取（JSON 持久化）
│   ├── visual/
│   │   ├── hand_tracker.py           # 步驟 3：MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── gesture_classifier.py     # 步驟 3：GRU 手勢意圖分類器（5 類）
│   │   ├── visual_scorer.py          # 步驟 3：視覺分數（OCR + SBERT + SRGAN）
│   │   ├── clip_scorer.py            # 步驟 3：CLIP 圖文跨模態對齊
│   │   ├── text_detector.py          # 步驟 3：文字區域偵測
│   │   ├── sbert_calculator.py       # 步驟 3：OCR-ASR 語意相似度
│   │   ├── srgan.py                  # 步驟 3：超解析度增強（模糊板書）
│   │   └── visual_reliability.py     # 步驟 3：視覺可靠度估測器
│   ├── fusion/
│   │   ├── adaptive_fusion.py        # 步驟 4：自適應融合 + Grid Search + LOOCV
│   │   ├── mlp_fusion.py             # 步驟 4：MLP 可學習融合模型
│   │   └── evaluator.py              # 效能評估（Recall/FAR/BERTScore/TSR/F1）
│   ├── output/
│   │   ├── video_exporter.py         # 步驟 5：FFmpeg 摘要影片與索引輸出
│   │   └── study_materials.py        # 步驟 5：學習素材產生器（筆記/心智圖/閃卡）
│   └── platform/
│       ├── api.py                    # FastAPI Web 後端
│       ├── celery_app.py             # Celery 非同步任務佇列
│       ├── task_store.py             # 本地任務狀態儲存（開發用）
│       └── static/index.html         # 前端介面
│
├── tools/
│   ├── fetch_archive_videos.py       # 從 Internet Archive 下載測試影片
│   ├── fetch_test_videos.py          # 從 YouTube 下載測試影片（yt-dlp）
│   ├── annotator.py                  # Ground Truth 雙盲標註工具
│   ├── export_transcript.py          # 批次逐字稿匯出
│   ├── run_grid_search.py            # Grid Search 最佳權重搜尋
│   ├── run_loocv.py                  # Leave-One-Out 交叉驗證
│   ├── run_batch_eval.py             # 批次效能評估
│   ├── run_domain_eval.py            # 跨場域泛化測試
│   ├── run_single_eval.py            # 單片評估
│   ├── run_demo_benchmark.py         # Demo 基準測試
│   ├── train_gesture_classifier.py   # GRU 手勢分類器訓練
│   ├── train_mlp_fusion.py           # MLP 融合模型訓練
│   ├── collect_gesture_data.py       # 手勢資料蒐集
│   ├── demo.py / diagnose.py         # 示範與診斷工具
│
├── data/
│   ├── prompt_corpus/corpus.py       # 教學提示語料庫（59 句）
│   ├── videos/                       # 測試影片（含 test_10min.mp4、downloaded/）
│   ├── annotations/                  # 標註資料與 Ground Truth
│   └── gesture_dataset*.json         # 手勢訓練資料集
│
├── weights/
│   ├── gesture_classifier.pth        # 訓練好的 GRU 手勢模型
│   └── mlp_fusion.pth                # 訓練好的 MLP 融合模型（可選）
│
├── models/
│   └── hand_landmarker.task          # MediaPipe 手部追蹤模型
│
├── tests/
│   ├── smoke_test.py                 # 煙霧測試（快速端到端）
│   ├── test_adaptive_fusion.py       # 融合與滑動視窗
│   ├── test_evaluator.py             # 評估指標
│   ├── test_gesture_classifier.py    # GRU 手勢分類
│   ├── test_semantic_scorer.py       # 語意評分
│   ├── test_text_detector.py         # 文字偵測
│   ├── test_visual_reliability.py    # 視覺可靠度
│   ├── test_sbert_calculator.py      # SBERT 計算（property-based，較慢）
│   └── test_visual_scorer.py         # 視覺分數（property-based，較慢）
│
└── results/                          # 評估報告輸出目錄
```

---

## 技術棧

| 類別 | 技術 | 說明 |
|------|------|------|
| 語音辨識 | OpenAI Whisper | 本地離線轉錄，支援遠端 API |
| 語意向量 | Sentence-BERT | 多語言語意相似度 |
| 大語言模型 | GPT-4o-mini / Groq Llama3 / Ollama | 教學結構分析、學習素材產生 |
| 手部追蹤 | MediaPipe Hands | 即時骨架偵測 + 卡爾曼濾波 |
| 手勢分類 | GRU 雙層網路 | 5 類教學意圖辨識 |
| 跨模態對齊 | OpenAI CLIP | 圖文語意對齊 |
| 韻律分析 | librosa | 語速 / 音量 / 音高 / 停頓 |
| OCR | Tesseract | 繁中 + 英文板書辨識 |
| 超解析度 | SRGAN（bicubic 降級） | 模糊板書增強 |
| 後端框架 | FastAPI + Celery + Redis | 非同步影片處理 |
| 評估指標 | BERTScore | 語意相似度評估 |
| 測試 | pytest + hypothesis | 單元 + property-based |

---

## 驗證結果

本專案於 **Python 3.14 / Windows** 完成以下驗證：

**測試套件**
- 核心單元測試 **106 項全數通過**（融合、評估、可靠度、手勢、語意、文字偵測）。
- 煙霧測試 **8/8 通過**（語意評分、手部追蹤、SRGAN 降級、融合、可靠度、Grid Search、評估、索引匯出）。
- property-based 測試（SBERT / 視覺分數）可正常 collect 與執行，因載入模型 + 大量隨機例子而耗時較長。

**真實影片端到端驗證**（透過 `tools/fetch_archive_videos.py` 從 Internet Archive 下載）

| 影片 | 原始時長 | 摘要時長 | 時間節省率 | 韻律分析 | 結果 |
|------|---------|---------|-----------|---------|------|
| College de France 講座 | 90.1s | 37.0s | 58.9% | ✓ | 選 2 段、手勢分布正常 |
| Archive 教學片 | 90.2s | 49.7s | 44.9% | ✓（0.446） | 選 3 段 |
| Archive 講課片 | 90.0s | 2.0s | 97.8% | ✓（0.500） | 選 1 段（低分內容，保證濃縮機制生效） |

四種模態（語意 / 視覺 / 韻律 / CLIP）、手勢分類器、學習素材產生皆確認實際運作，
且在缺少選填套件時能優雅降級。

> **重現方式**：
> ```bash
> cd keytake-ai
> python tools/fetch_archive_videos.py --count 3 --clip-seconds 90
> python main.py "data/videos/downloaded/<檔名>.mp4"
> ```

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

### Q：沒有 GPU 可以跑嗎？
可以。所有模組都支援 CPU。Whisper 建議用 `base` 模型，一部 50 分鐘課程約需 5~10 分鐘。

### Q：不想付 LLM API 費用怎麼辦？
三種免費方案：
1. `SEMANTIC_SCORER_MODE=sbert` — 本地 SBERT（不需 API）
2. `LLM_BACKEND=groq` + 免費 Groq API key — 免費額度通常夠用
3. `LLM_BACKEND=ollama` + 本地 Ollama — 完全離線免費

### Q：librosa / transformers / groq 沒裝會怎樣？
系統自動降級，不會崩潰：
- librosa 未裝 → 韻律分數固定 0.5
- transformers 未裝 → 跳過 CLIP，只用 OCR→SBERT
- groq 未裝 → Groq 後端不可用（可改用 sbert / openai / ollama）
- sentence-transformers 未裝 → 語意評分自動降級為 TF-IDF

### Q：為什麼有的影片摘要非常短（例如只留 2 秒）？
當整部影片的融合分數普遍偏低（內容平淡、無明顯重點/板書），
門檻選段後只有少數片段勝出，加上「保證濃縮」上限修剪，就會產生很短的摘要。
這是預期行為；可透過調低 `FUSION_SCORE_THRESHOLD` 或提高 `MAX_SUMMARY_RATIO` 保留更多片段。

### Q：Tesseract 找不到？
確認已安裝 Tesseract 本體，並在 `.env` 設 `TESSERACT_PATH` 指向 `tesseract.exe`，
或把安裝目錄加入系統 PATH。未安裝時視覺模組會走降級路徑（IoU 備援）。

### Q：怎麼快速確認系統正常？
```bash
python tests/smoke_test.py
```
看到 `所有測試通過 ✓` 即代表核心串接正常。

---

## 授權

本專案為學術研究用途。
