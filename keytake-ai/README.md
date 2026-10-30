# KeyTake AI — 程式碼說明

本目錄包含 KeyTake AI 系統的完整程式碼。
研究背景、系統架構與實驗結果請參閱[專案根目錄 README](../README.md)。

---

## 環境需求

- Python 3.10 以上（建議使用 Python 3.12；本工作區以 Python 3.14.3 驗證）
- 外部工具（需另外安裝，**不透過 pip**）：
  - [FFmpeg](https://ffmpeg.org/download.html)：影音轉換與音訊提取
  - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)：板書文字辨識（Windows 請用 UB-Mannheim 版本，安裝時記得勾選中文語言包）

---

## 安裝

```bash
cd keytake-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 PowerShell 複製環境變數範本（若 `.env` 尚不存在）：

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

`.env` 預設設定（不需要 API Key 即可執行）：

```
SEMANTIC_SCORER_MODE=sbert
WHISPER_MODE=local
```

若要啟用 LLM 語意評分（會依所選 backend 呼叫外部服務或本機 Ollama；需按資料政策設定），將 `.env` 改為：

```
SEMANTIC_SCORER_MODE=llm
LLM_BACKEND=groq          # 選擇已設定憑證且符合資料政策的服務
GROQ_API_KEY=gsk_xxx
```

`SEMANTIC_SCORER_MODE` 會明確決定評分路徑；設定 `sbert` 時，即使 `.env` 有 LLM Key 也不會呼叫遠端 LLM。
學習素材生成預設跟隨此模式；只有明確設定 `STUDY_MATERIALS_MODE=llm` 才會單獨啟用遠端生成。
本地 Whisper 的 `base` 權重會在首次轉錄時由 Whisper 下載；影片處理需要可執行的 FFmpeg。OCR 需安裝 Tesseract，並將其加入 `PATH` 或設定 `TESSERACT_PATH`。

---

## 快速開始

### 命令列模式

```powershell
# 基本用法：輸出摘要至 output/ 目錄
python main.py "data/videos/lecture_01_math_homework.mp4"

# 搭配 Ground Truth 評估效能（Recall / FAR / TSR）
python main.py "data/videos/lecture_01_math_homework.mp4" --gt "data/annotations/gt_lecture_01_math_homework.json"
```

### Web 介面模式

前端由 FastAPI 提供，不需另外安裝 Node.js。Windows 開發模式需要 Redis、Celery worker 與 FastAPI 三個程序：

```powershell
# 1. 啟動 Redis（需先安裝並啟動 Docker Desktop）
docker run --name keytake-redis -d --restart unless-stopped -p 6379:6379 redis:7-alpine

# 2. 在另一個 PowerShell 視窗，進入 keytake-ai 並啟用 venv 後啟動 worker
celery -A src.platform.celery_app:celery_app worker --pool=solo --loglevel=info

# 3. 在第三個 PowerShell 視窗，進入 keytake-ai 並啟用 venv 後啟動 API
python -m uvicorn src.platform.api:app --reload --host 127.0.0.1 --port 8000
```

開啟 `http://127.0.0.1:8000` 使用介面；`http://127.0.0.1:8000/health` 應回傳 `{"status":"ok"}`。
若不使用 Redis/Celery，可在 API 視窗啟動前設定 `$env:KEYTAKE_SYNC_FALLBACK="1"`，由 API process 的背景執行緒處理（僅適合單機開發）。

停止服務時，在 worker/API 視窗按 `Ctrl+C`；停止 Redis 容器可執行 `docker stop keytake-redis`。

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
│   │   └── preprocessor_optimized.py # FFmpeg 轉換、音訊提取、SNR 降噪
│   │
│   ├── semantic/
│   │   ├── transcriber_optimized.py # Whisper 語音轉錄（本地/遠端兩模式）
│   │   ├── semantic_scorer_optimized.py # 語意評分（SBERT / LLM / TF-IDF）
│   │   ├── prosodic_analyzer_optimized.py # 韻律特徵
│   │   └── llm_cache_optimized.py # LLM 回應快取
│   │
│   ├── visual/
│   │   ├── hand_tracker.py     # MediaPipe 手部追蹤 + 卡爾曼濾波
│   │   ├── gesture_classifier.py # 雙層 GRU 手勢意圖分類
│   │   ├── visual_scorer.py    # OCR → SBERT 語意比對、IoU 備援
│   │   ├── clip_scorer.py      # CLIP 板書影像與口語關鍵字對齊
│   │   ├── visual_reliability.py # 四訊號加權估計視覺可靠度 R_visual
│   │   ├── text_detector.py    # pytesseract 文字區域偵測
│   │   ├── sbert_calculator.py # OCR 文字與 ASR 關鍵字語意相似度
│   │   └── srgan.py            # 模糊偵測 + OpenCV bicubic resize（非 SRGAN 模型）
│   │
│   ├── fusion/
│   │   ├── adaptive_fusion.py  # 自適應融合、語意感知滑動視窗
│   │   ├── evaluator.py        # Recall / Precision / F1 / FAR / TSR
│   │   ├── feature_extractor.py# MLP 融合用特徵擷取
│   │   └── mlp_fusion.py       # 可學習 MLP 融合模型（實驗性）
│   │
│   ├── output/
│   │   ├── video_exporter_optimized.py # 剪輯輸出摘要影片、精華索引
│   │   └── study_materials_generator_optimized.py # 筆記、心智圖、閃卡
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
│   ├── extract_features.py     # 擷取 MLP 訓練所需的完整特徵
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
│   ├── test_hand_tracker.py
│   ├── test_sbert_calculator.py
│   ├── test_semantic_scorer.py
│   ├── test_text_detector.py
│   ├── test_transcriber.py
│   ├── test_clip_scorer.py
│   ├── test_prosodic_analyzer.py
│   ├── test_mlp_fusion.py
│   ├── test_visual_reliability.py
│   ├── test_visual_scorer.py
│   └── test_video_exporter.py
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
│   └── mlp_fusion_12d.pth      # MLP 訓練後的本機實驗權重（不隨程式碼提交）
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

根目錄 README 保留了先前對 `test_10min` 的 β 掃描數值，作為歷史實驗紀錄。該影片不能再視為完全 untouched 的 holdout，也不應重跑此命令來選參；目前不要使用 `test_10min` 反覆調整任何權重。

### 在明確指定的 validation 影片上搜尋融合權重

```powershell
python tools/run_grid_search.py --features-dir results/features_12d --annotations data/annotations --validation-video lecture_01_math_homework --validation-video lecture_02_trigonometry --output results/grid_search.json
```

此命令只讀取明確列出的 validation 影片，預設按實際視覺可靠度搜尋 Adaptive Fusion 權重；`--fusion-mode static` 可供靜態基準比較。只可列入訓練/驗證影片，不要把 `test_10min` 列入，也不要將小樣本候選描述成正式最佳設定。

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

### 訓練 MLP 融合模型

先對每部有對應 Ground Truth 的影片抽取完整特徵，再執行訓練：

```powershell
New-Item -ItemType Directory -Force results\features_12d | Out-Null
$env:SEMANTIC_SCORER_MODE = "sbert"
$env:WHISPER_MODE = "local"
python tools/extract_features.py --video data/videos/lecture_01_math_homework.mp4 --output results/features_12d/features_lecture_01_math_homework.json
python tools/extract_features.py --video data/videos/lecture_02_trigonometry.mp4 --output results/features_12d/features_lecture_02_trigonometry.json
python tools/extract_features.py --video data/videos/lecture_03_mental_math.mp4 --output results/features_12d/features_lecture_03_mental_math.json
python tools/extract_features.py --video data/videos/test_10min.mp4 --output results/features_12d/features_test_10min.json
python tools/train_mlp_fusion.py --features results/features_12d --annotations data/annotations --test-features results/features_12d/features_test_10min.json --test-annotations data/annotations/gt_test_10min.json --output weights/mlp_fusion_12d.pth
```

`SEMANTIC_SCORER_MODE=sbert` 與 `WHISPER_MODE=local` 會讓特徵抽取使用本機語意模型和本機 Whisper，避免將影片或轉錄資料送至遠端 API。新 checkpoint 使用獨立的 `mlp_fusion_12d.pth` 路徑，不覆寫舊模型。

訓練程式會從訓練資料夾排除明確指定的獨立測試影片；測試特徵和標註必須一一對應，且不參與訓練或 validation 選擇。訓練與測試都會依 Ground Truth 的 `eval_range_sec` 裁切／平移特徵和標註，避免未標註影片範圍影響評估。若使用 test_10min 作獨立測試，不能再把它混入訓練資料。
注意：`test_10min` 曾被較早版本用於 β 敏感度掃描；近期 12 維 MLP 對它的 F1=0.500 是一次新評估結果，但不會使它成為從未使用過的測試集。不得再用該影片調參或宣稱完全未觸碰的 holdout。
特徵 JSON 必須含 `feature_version`，且版本符合目前 12 維特徵 schema；schema 改版後需重新抽取，舊特徵檔會被拒絕。`prosodic_available` 會標示韻律分數是否來自有效音訊分析；缺值使用中性填充值時，availability mask 仍會讓 MLP 知道該分數不可用。
checkpoint 會保存 `model_state_dict`、輸入維度、特徵版本/順序、教學階段及標籤規則；載入不相容的權重會明確退回 Adaptive Fusion，不會靜默套用。
`sbert`/`tfidf` 評分模式不推斷教學階段，片段會標為 `unclassified`，不會假裝成 `transition`。MLP 訓練至少需要兩部影片，驗證集會按影片分組，避免同一影片的相鄰片段同時出現在訓練與驗證資料。
MLP 正樣本預設依 `overlap / Ground Truth duration >= 0.5` 定義，與 evaluator 的重點召回率標準一致；可用 `--label-overlap-basis segment` 重現舊的片段覆蓋規則，但此規則與評估器不同，結果應明確標註。
視覺可靠度與韻律語速特徵的定義已更新；請重新抽取所有特徵、重訓並完成獨立評估後，才將 `config.py` 的 `USE_MLP_FUSION` 設為 `True`。目前預設停用舊權重，避免將舊特徵語義套用到新輸入。

視覺可靠度只在對應視覺分數可用時生效；缺少可靠度欄位會按不可用處理。CLIP cosine similarity 不是機率或教學重要性分數；音訊 `snr` 也是啟用條件式降噪的啟發式能量比，需在具代表性的音訊與轉錄品質上驗證，不能當成實測 SNR。

摘要影片的片段索引使用原片時間；播放摘要影片時，前端會改用匯出的 source/output timeline mapping 定位片段。若沒有時間對照資料，摘要播放按鈕會保持停用，避免將原片時間錯當成摘要影片時間。

Grid Search 會從 `features_<video>.json` 載入完整多模態特徵，逐影片套用相同的摘要選段流程，以宏平均 F1 為主要目標，並同時回報 Recall/Precision/F1；LOOCV 同樣使用完整特徵，訓練影片與留出影片分開評估。兩者會依 GT 的 `eval_range_sec` 裁切並平移特徵/標註時間軸，避免評估範圍外的片段超出摘要預算。`semantic_sliding_window` 會依原片時長（主流程可提供）限制摘要預算，拒絕長度不一致或無效的片段/分數輸入。預算超限時預設保留舊版分數優先策略；消融工具可用 `--budget-strategy score_per_second` 與預設 `score` 配對比較，尚未經獨立測試集驗證前不宣稱新策略摘要品質較佳。

```powershell
python tools/run_grid_search.py --features-dir results/features_12d --annotations data/annotations --validation-video lecture_01_math_homework --validation-video lecture_02_trigonometry
python tools/run_loocv.py --features-dir results/features_12d --annotations data/annotations --video lecture_01_math_homework --video lecture_02_trigonometry --video lecture_03_mental_math
python tools/run_ablation_batch.py --use-cached-features --video lecture_01_math_homework --video lecture_02_trigonometry --video lecture_03_mental_math --budget-strategy score_per_second --output results/ablation_batch_score_per_second.json
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
| `ALPHA` | `0.6` | 目前實驗基準；三部訓練影片搜尋候選 0.5/0.5 未正式採用 |
| `BETA` | `0.4` | 目前視覺基準權重；不是已驗證最佳值 |
| `FUSION_SCORE_THRESHOLD` | `0.2` | 片段最低融合分數 |
| `MAX_SUMMARY_RATIO` | `0.6` | 摘要最長為原片的 60% |
| `GESTURE_INTENT_THRESHOLD` | `0.7` | 觸發視覺分析的最低意圖分數 |
| `CLIP_WEIGHT` | `0.3` | CLIP 在視覺分數中的混合比例 |
| `PROSODIC_WEIGHT` | `0.15` | 韻律特徵在融合中的影響權重 |

三部訓練影片的 α/β 搜尋候選為 0.50/0.50、macro F1=0.5341；`prosodic_weight_sweep_12d_training_only.json` 中的 0.30、macro F1=0.5611 是**韻律權重**探索，不是 CLIP sweep。原始小型 JSON 紀錄見 [`grid_search_12d_training_only.json`](results/grid_search_12d_training_only.json) 與 [`prosodic_weight_sweep_12d_training_only.json`](results/prosodic_weight_sweep_12d_training_only.json)。兩者皆屬小樣本結果；`CLIP_WEIGHT=0.3` 是現行程式設定，沒有上述 sweep 的結果支持。完整模組流程及口試教材見[系統技術說明書](../docs/KeyTake%20AI%20系統架構與技術實作說明書.md)與[口試教學文件](../docs/KeyTake%20AI%20研究所口試教學與問答.md)。

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
