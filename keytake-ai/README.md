# KeyTake AI — 語意-視覺融合的教育影片自動摘要系統

> A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization

KeyTake AI 是一套針對「教學影片」設計的自動摘要系統。它同時分析**聲音（老師說什麼）**、**畫面（老師寫/指什麼）**、**手勢意圖（老師在強調什麼）**與**語調韻律（老師怎麼說）**，找出一堂課裡真正的重點時段，剪成一部濃縮影片，並自動產生章節、Markdown 筆記、心智圖與問答閃卡等學習素材。

本文件完整記錄系統的**設計理念、每一個處理步驟的實作方法、演算法細節、參數設定、評估流程與使用方式**，可直接作為論文方法章節與競賽報告的技術依據。

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [整體架構與資料流](#2-整體架構與資料流)
3. [五大處理步驟的詳細作法](#3-五大處理步驟的詳細作法)
4. [核心創新點](#4-核心創新點)
5. [評估方法與工具](#5-評估方法與工具)
6. [安裝與使用](#6-安裝與使用)
7. [設定參數對照表](#7-設定參數對照表)
8. [目錄結構](#8-目錄結構)
9. [測試與品質保證](#9-測試與品質保證)
10. [已知限制與選用進階功能](#10-已知限制與選用進階功能)

---

## 1. 系統總覽

傳統影片摘要多半只看單一模態（例如只做語音關鍵字），在教學場景會漏掉「老師在黑板上寫的推導」或「投影片上的公式」。KeyTake AI 的核心主張是：**教學重點通常同時出現在「講述」與「板書/投影片」上**，因此用多模態融合能更準確地抓到重點，並在困難場景（反光、低對比）維持穩定。

系統特色：

- **四模態融合**：語意（S_text）× 視覺（S_visual）× 手勢意圖 × 韻律（S_prosodic）。
- **自適應置信度加權**：視覺不可靠時自動降低視覺權重，避免雜訊污染摘要（v2 創新）。
- **優雅降級設計**：缺少 GPU、API Key、Tesseract、librosa 等任一元件時，對應模組自動退回較簡單的替代方案，系統不會崩潰。
- **多元輸出**：不只剪影片，還產生章節、筆記、心智圖、閃卡。
- **完整評估管線**：Grid Search、留一交叉驗證（LOOCV）、跨場域評估、批次評估。

---

## 2. 整體架構與資料流

```
                        ┌──────────────────────────────┐
   教學影片 (mp4/webm)  │        Step 1 前處理          │
        ─────────────► │  FFmpeg 轉檔 + 抽音 + SNR 判斷 │
                        │  (低 SNR 才做頻譜減法降噪)     │
                        └──────────────┬───────────────┘
                                       │ audio.wav / video.mp4
             ┌─────────────────────────┼─────────────────────────┐
             ▼                         ▼                         ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │  Step 2 語意分析     │  │ Step 2.5 韻律分析   │  │  Step 3 視覺分析     │
  │  Whisper 轉錄        │  │ 語速/音量/音高/停頓 │  │ 每段取 3 幀          │
  │  → 幻覺去重          │  │ → S_prosodic        │  │ MediaPipe 手部追蹤   │
  │  LLM/SBERT/TFIDF評分 │  └────────────────────┘  │ + Kalman 平滑        │
  │  → S_text            │                          │ + GRU 手勢意圖分類   │
  │  + 教學階段/摘要句   │                          │ OCR→SBERT / CLIP     │
  └──────────┬───────────┘                          │ 模糊→SRGAN 增強      │
             │                                       │ → S_visual + 可靠度  │
             │                                       └──────────┬───────────┘
             └───────────────────┬──────────────────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │      Step 4 多模態融合          │
                  │  β_eff = β × R_visual (自適應)  │
                  │  Score = α_eff·S_text +         │
                  │          β_eff·S_visual         │
                  │  × 韻律加權 (或 MLP 融合)       │
                  │  → 語意感知滑動視窗選段         │
                  └────────────────┬───────────────┘
                                   ▼
        ┌───────────────────────────────────────────────┐
        │   輸出：摘要影片 + 章節 + 筆記 + 心智圖 + 閃卡  │
        │   評估：Recall / FAR / TSR / BERTScore          │
        └───────────────────────────────────────────────┘
```

主流程進入點：`main.py::run_pipeline(video_path, output_dir, progress_callback, ground_truth)`
平台層進入點：`src/platform/api.py`（FastAPI + Celery 非同步任務）

---

## 3. 五大處理步驟的詳細作法

### Step 1｜影音前處理 — `src/preprocessing/preprocessor.py`

目的：把任意格式影片正規化，並準備乾淨的音訊供轉錄。

作法：
1. **格式正規化**：用 FFmpeg 將輸入影片統一轉為 30fps 的 MP4，確保後續逐幀取樣的時間戳一致。
2. **音訊抽取**：抽出單聲道、16kHz 的 WAV（Whisper 的建議輸入規格）。
3. **雜訊判斷與降噪**：估算音訊訊噪比（SNR）。只有當 `SNR < AUDIO_SNR_THRESHOLD (=10dB)` 時才啟用**頻譜減法（Spectral Subtraction）**降噪，避免對本來就乾淨的錄音過度處理而損失語音細節。
4. 回傳 `{"video": ..., "audio": ..., "snr": ...}`。

設計理由：條件式降噪是為了在「錄音品質差的教室」與「品質好的線上課」之間取得平衡，而非一律套用降噪。

---

### Step 2｜語音轉錄 + 語意評分 — `src/semantic/`

**2-1 轉錄 `transcriber.py::transcribe()`**
- 支援 `local`（本機 Whisper）與 `remote`（OpenAI Whisper API，快約 10 倍）兩種模式，由 `.env` 的 `WHISPER_MODE` 控制。
- **幻覺去重（`_deduplicate_segments`）**：Whisper 在靜音段常吐出重複的幻覺字句（例如反覆的「謝謝觀看」），這裡會偵測並移除連續重複片段，提升後續評分品質。
- 輸出帶時間戳的片段清單 `[{start, end, text}, ...]`。

**2-2 語意評分 `scorer.py::SemanticScorer.score()`**
提供三軌評分策略（由 `SEMANTIC_SCORER_MODE` 選擇），皆輸出每段的重要性分數 `s_text ∈ [0,1]`：
- **`llm`**：呼叫大語言模型（OpenAI / Groq / Ollama），對每段輸出「重要性分數 + 教學階段（導入/講解/舉例/總結…）+ 一句摘要」。效果最好，並支援全課脈絡分析。
- **`sbert`**：本地 Sentence-BERT，計算每段文字與「教學重點提示語料庫」（`data/prompt_corpus/`）的最大餘弦相似度，離線可用，為預設值。
- **`tfidf`**：純統計 TF-IDF 關鍵詞權重，最快、零外部依賴。

**2-3 LLM 快取 `llm_cache.py`**：以 SHA256 對輸入做鍵值快取（含 TTL、執行緒安全、批次寫入），避免重複 API 呼叫，讓結果穩定且省費用。

**2-4 全課分析 `generate_course_summary()`**：彙整所有片段產出整堂課的標題與摘要，供前端顯示與章節命名。

---

### Step 2.5｜韻律特徵分析 — `src/semantic/prosodic_analyzer.py`（v5）

洞見：老師強調重點時，語調會改變 —— 放慢語速、提高音量、音高起伏變大、前後有明顯停頓。

作法：用 librosa 從音訊逐段提取四項韻律特徵並加權合成 `s_prosodic`：
- 語速（`PROSODIC_SPEECH_RATE_WEIGHT=0.3`）
- 音量（`PROSODIC_VOLUME_WEIGHT=0.3`）
- 音高變化（`PROSODIC_PITCH_WEIGHT=0.25`）
- 停頓比例（`PROSODIC_PAUSE_WEIGHT=0.15`）

降級：未安裝 librosa 時，每段回傳中性值 0.5（等於不影響融合）。

---

### Step 3｜視覺特徵提取 — `src/visual/`

這是系統最複雜的部分。對每個語意片段，取其 **25% / 50% / 75% 三個時間點**的畫面（多幀取樣，降低單幀誤判），逐幀計算視覺重要性，取三幀最高分作為該段代表。

**3-1 手部追蹤與意圖分類 `hand_tracker.py` + `gesture_classifier.py`**
- **MediaPipe** 偵測手部骨架關鍵點。
- **Kalman 濾波**平滑手部軌跡，抑制抖動。
- **GRU 手勢意圖分類器**（權重 `weights/gesture_classifier.pth`）：把連續 `GESTURE_SEQ_LEN=20` 幀的軌跡序列送入 GRU，分類為 5 種教學意圖（如「書寫」「指示」「無意義移動」等），每種意圖對應不同重要性。意圖分數低於 `GESTURE_INTENT_THRESHOLD=0.7` 的片段視覺權重會被壓低。缺模型檔時退回規則型判斷。

**3-2 文字偵測與語意比對 `text_detector.py` + `sbert_calculator.py`**
- 在手部附近裁切 `ROI_SIZE=224` 的感興趣區域。
- **Tesseract OCR** 辨識 ROI 內的板書/投影片文字。
- 用 **SBERT** 計算「OCR 文字」與「該段語音關鍵字」的語意相似度 —— 講的和寫的一致時得分高，代表這是真正的教學重點。
- **降級備援**：OCR 失敗或無 SBERT 時，改用手部框與文字框的 **IoU 重疊**（`IOU_FALLBACK_THRESHOLD=0.3`）作為替代訊號。

**3-3 CLIP 跨模態對齊 `clip_scorer.py`（v5）**
- 用 CLIP 直接做「圖片 ↔ 文字」語意匹配，**跳過 OCR**。這在 OCR 難以辨識的困難場景（手寫潦草、反光）特別有用。
- 與 OCR→SBERT 路徑**加權混合**：`combined = (1-CLIP_WEIGHT)·OCR分數 + CLIP_WEIGHT·CLIP分數`（`CLIP_WEIGHT=0.3`）。以 Singleton 載入避免重複佔用記憶體。

**3-4 模糊偵測與超解析度增強 `srgan.py`**
- 用 **Laplacian 變異數**判斷 ROI 是否模糊（`BLUR_THRESHOLD=100`）。
- **只有模糊時才觸發增強**（`enhance_if_blurry`），節省運算。目前預設用 OpenCV bicubic 放大（`SRGAN_SCALE=4`）；Real-ESRGAN 為選用延伸（見第 10 節）。

**3-5 視覺可靠度估測 `visual_reliability.py`（v2 核心）**
- 綜合「手是否在板書前」「文字框密度」「手部穩定度」等可觀測訊號，估算視覺可靠度 `R_visual ∈ [0,1]`，作為下一步自適應融合的關鍵輸入。

輸出：每段的 `s_visual`、`visual_reliability`、`gesture_label`。

---

### Step 4｜多模態融合與動態剪輯 — `src/fusion/`

**4-1 自適應置信度加權融合 `adaptive_fusion.py::fuse_scores()`（v2 核心創新）**

從靜態加權升級為自適應：
```
β_eff = β × R_visual          # 視覺不可靠 → 自動降權
α_eff = 1 − β_eff             # 語意權重自動補足，總和恆為 1
Score = α_eff·S_text + β_eff·S_visual
```
- 手勢不在板書前（R_visual 低）→ 自動提高語意權重，避免視覺雜訊污染摘要。
- 板書密度高且手部穩定（R_visual 高）→ 視覺權重提升，充分利用畫面資訊。
- 當 `R_visual = 1.0` 時退化為傳統靜態 α/β，向後相容。

**4-2 韻律加權**：融合後再依 `s_prosodic` 做微調 —— 韻律分數 0.5 無影響，高於 0.5 提升、低於 0.5 壓低，權重 `PROSODIC_WEIGHT=0.15`。

**4-3 MLP 可學習融合 `mlp_fusion.py`（v5，選用）**：`USE_MLP_FUSION=True` 時改用監督式 MLP 取代固定公式；模型不存在時自動退回解析式公式。

**4-4 語意感知滑動視窗 `semantic_sliding_window()`**：這是剪輯的關鍵。系統不是只剪出高分的那一秒，而是以高分點為錨，依 Whisper 斷句向前後延伸，確保「板書推導」與「一句完整的話」不會被切斷。最小保留時長 `SLIDING_WINDOW_MIN_SEC=10` 秒。保留門檻 `FUSION_SCORE_THRESHOLD=0.2`。

**4-5 保證濃縮機制（不論影片長度，輸出必定比原片短）**：滑動視窗內建三層保證，解決「短影片或全高分影片幾乎沒被縮短」的問題：
1. **總長度上限**：摘要總時長不得超過「原片 × `MAX_SUMMARY_RATIO`（預設 0.6）」。若門檻選段結果超出上限，依融合分數由高到低貪婪保留片段，直到逼近上限——確保至少節省 40%。
2. **短影片保護**：當「原片 × 上限」比最小片段長度還短時，自動縮小最小片段長度，避免單一片段就吃掉整部片。
3. **保底輸出**：即使全部片段都低於門檻，也會保留分數最高的一段（截斷至上限內），確保任何影片都有可看的濃縮輸出，且嚴格短於原片。

此性質以 property-based 測試（200 組隨機長度/分數）驗證：`summary_duration < original_duration` 恆成立。

---

### Step 5｜輸出與學習素材 — `src/output/`

- **`video_exporter.py`**：用 FFmpeg 依選中的時間段剪輯並串接出摘要影片，並產生索引（每段的標籤用 LLM 摘要句）。
- **`study_materials.py::StudyMaterialsGenerator.export_all()`**：一次產出
  - `chapters.json`（章節標題與時間）
  - `study_notes.md`（Markdown 學習筆記）
  - `mind_map.json`（心智圖）
  - `flashcards.json`（問答閃卡）
- 學習素材產生失敗時不影響主摘要流程（try/except 隔離）。

---

## 4. 核心創新點

適合寫入論文「貢獻」與競賽「亮點」章節：

1. **自適應置信度加權晚期融合**：以可觀測訊號估算視覺可靠度，動態調整模態權重（Uncertainty-aware Multimodal Fusion），解決固定權重在困難場景失效的問題。
2. **四模態融合**：在語意+視覺之外，額外引入「手勢意圖」與「語調韻律」兩個教學專屬訊號。
3. **雙路徑視覺理解**：OCR→SBERT（精確）與 CLIP（強健）互補混合，兼顧準確與抗噪。
4. **語意感知滑動視窗**：以語言邊界而非固定時窗剪輯，保持推導與語句完整性。
5. **全鏈路優雅降級**：任一重量元件缺失都能退回替代方案，兼顧研究完整性與部署可行性。

---

## 5. 評估方法與工具（`tools/`）

| 工具 | 用途 |
|------|------|
| `annotator.py` | Ground Truth 標註 CLI，支援 Likert 1–5、雙標註者合併、Cohen's Kappa 一致性檢驗（>0.6） |
| `export_transcript.py` | 匯出逐字稿與語意分數，供標註與調參 |
| `run_grid_search.py` | Grid Search 搜尋最佳 α/β |
| `run_loocv.py` | 留一交叉驗證（Leave-One-Out），驗證泛化性 |
| `run_batch_eval.py` | 多影片批次評估（Recall / FAR / TSR / BERTScore） |
| `run_domain_eval.py` | 跨場域評估（黑板 / 投影片 / 高反光），輸出 Markdown 報告 |
| `run_demo_benchmark.py` | 蒙地卡羅**模擬**基準（無真實影片時展示預期效能，報告內明確標註「需真實資料驗證」） |
| `train_gesture_classifier.py` / `train_mlp_fusion.py` | 訓練 GRU 手勢分類器 / MLP 融合模型 |
| `collect_gesture_data.py` / `run_gesture_pipeline.py` | 蒐集手勢資料 / 一鍵手勢訓練流程 |
| `demo.py` | 一鍵示範（用範例資料，不需影片/GPU/API） |
| `diagnose.py` | 環境依賴診斷 |

**評估指標**（`src/fusion/evaluator.py`）：重點召回率（Recall，重疊 >50% 視為命中）、誤報率（FAR）、時間節省率（TSR）、精確率、F1、BERTScore。目標值定義於 `config.py`：Recall≥0.70、FAR<0.25、BERTScore≥0.70、TSR≥0.50。

> ⚠️ **重要（論文誠信）**：`results/benchmark_demo.json` 是**模擬數據**，用於在缺乏標註資料集時展示架構的效能預期。論文與競賽正式數據必須用真實教學影片，透過 `annotator.py` 標註 GT 後以 `run_batch_eval.py` / `run_domain_eval.py` 產生。

---

## 6. 安裝與使用

### 6-1 系統需求（非 pip，需另裝）
- **FFmpeg**：影音轉檔（系統會自動偵測常見安裝路徑）
- **Tesseract OCR**：板書文字辨識（Windows 版見 UB-Mannheim wiki）
- Python 3.10+（建議）

### 6-2 安裝
```bash
pip install -r requirements.txt
copy .env.example .env     # 依需求填入 API Key 與模式
```

### 6-3 執行單部影片（CLI）
```bash
# 基本用法
python main.py "你的教學影片.mp4"

# 附上 Ground Truth 以計算 Recall / FAR
python main.py "lecture.mp4" --gt data/annotations/lecture_gt.json
```

### 6-4 啟動 Web 平台
```bash
uvicorn src.platform.api:app --reload
# 開啟瀏覽器上傳影片；長影片透過 Celery 非同步處理
# （未啟動 Redis/Celery 時自動降級為 threading 同步執行）
```

### 6-5 一鍵示範（不需影片/GPU/API）
```bash
python tools/demo.py
```

---

## 7. 設定參數對照表（`config.py` 摘要）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `WHISPER_MODEL` | `base` | Whisper 模型大小 |
| `SBERT_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 支援中文的 SBERT |
| `ALPHA` / `BETA` | `0.6` / `0.4` | 語意/視覺靜態權重（Grid Search 調校） |
| `FUSION_SCORE_THRESHOLD` | `0.2` | 保留片段的最低融合分數 |
| `SLIDING_WINDOW_MIN_SEC` | `10` | 滑動視窗最小保留秒數 |
| `MAX_SUMMARY_RATIO` | `0.6` | 摘要總時長上限比例（保證至少節省 40%） |
| `MIN_SUMMARY_SEC` | `2.0` | 摘要最短輸出秒數（保底輸出） |
| `VISUAL_SAMPLE_FPS` | `1` | 視覺分析取樣頻率 |
| `VISUAL_FRAMES_PER_SEGMENT` | `3` | 每段取樣幀數（25/50/75%） |
| `BLUR_THRESHOLD` | `100` | 模糊偵測閾值 |
| `GESTURE_INTENT_THRESHOLD` | `0.7` | 觸發視覺分析的最低意圖分數 |
| `PROSODIC_WEIGHT` | `0.15` | 韻律在融合中的權重 |
| `CLIP_WEIGHT` | `0.3` | CLIP 分數在視覺分數中的混合權重 |
| `USE_MLP_FUSION` | `False` | 是否啟用 MLP 融合（需先訓練） |

---

## 8. 目錄結構

```
keytake-ai/
├── main.py                     # 主流程進入點 run_pipeline()
├── config.py                   # 全域參數 + FFmpeg/Tesseract 自動偵測
├── requirements.txt
├── .env.example                # 環境變數範本
├── src/
│   ├── preprocessing/          # Step 1 前處理（轉檔/抽音/降噪）
│   ├── semantic/               # Step 2 轉錄/評分/韻律/LLM 快取
│   ├── visual/                 # Step 3 手部/OCR/SBERT/CLIP/SRGAN/可靠度
│   ├── fusion/                 # Step 4 自適應融合/評估/MLP 融合
│   ├── output/                 # Step 5 影片匯出/學習素材
│   └── platform/               # FastAPI + Celery + 前端 static
├── tools/                      # 標註/評估/訓練/示範腳本
├── tests/                      # 單元測試 + property-based + 煙霧測試
├── data/                       # 語料庫/測試影片/GT 標註
├── weights/                    # 訓練好的模型權重（GRU 手勢分類器）
├── models/                     # MediaPipe 等外部模型
└── results/                    # 評估結果 JSON / 報告
```

---

## 9. 測試與品質保證

- **156 個測試**，涵蓋 evaluator、adaptive_fusion、gesture_classifier、sbert_calculator、semantic_scorer、text_detector、visual_reliability、visual_scorer 及整合煙霧測試。
- **Property-based testing（Hypothesis）**：融合、評估、SBERT、可靠度、視覺評分等模組使用 `@given` 隨機性質測試，確保在大量隨機輸入下數學性質（分數落在 [0,1]、權重和為 1 等）恆成立。
- 執行：
```bash
python -m pytest tests/ -v
# 只跑輕量單元測試（避免載入重模型）：
python -m pytest tests/test_adaptive_fusion.py tests/test_evaluator.py -v
```

> 注意：完整 pytest 會載入 SBERT/Whisper/CLIP 等重模型，首次執行較慢。

---

## 10. 已知限制與選用進階功能

以下項目**皆有 fallback，不影響系統正常運作**，屬設計上的選用延伸：

1. **Real-ESRGAN 超解析度**：`srgan.py` 已寫好載入介面，但預設用 OpenCV bicubic 放大取代（避免大型權重依賴）。需要時可手動啟用 Real-ESRGAN。
2. **MLP 融合預設關閉**：`USE_MLP_FUSION=False`；啟用需先用 `train_mlp_fusion.py` 訓練 `weights/mlp_fusion.pth`，未訓練時自動退回解析式融合公式。
3. **正式評估數據需真實影片**：`benchmark_demo.json` 為模擬結果，論文/競賽數據請用真實影片走完整標註與評估流程。

---

## 授權與引用

若本系統用於學術發表，請引用專案標題：
*A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization*。
