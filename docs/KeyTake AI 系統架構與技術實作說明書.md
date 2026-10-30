# KeyTake AI 系統架構與技術實作說明書

> 版本依據：目前工作目錄中的程式碼與 12 維特徵實驗。本文記錄實際呼叫路徑；README、舊實驗報告或簡報若與程式不一致，以程式碼為準。
> 研究限制：目前人工 Ground Truth 是單人標註 pilot；四部影片的數量不足以證明泛化能力。文中的「分數」多為排序特徵，不等同機率或已驗證的教學品質。

## 1. 系統概覽

### 1.1 架構圖

```text
使用者
  │ 選擇影片、瀏覽上傳進度、播放原片／摘要
  ▼
單頁前端：src/platform/static/index.html
  │ POST /upload（multipart/form-data）
  ▼
FastAPI：src/platform/api.py → keytake_api_optimized.py
  ├─ 驗證副檔名、分塊寫入 tmp/uploads/<task_uuid>.<ext>、限制 500 MiB
  ├─ 預設檢查 Redis，將任務交給 Celery；開發者可選同步執行緒降級模式
  └─ GET /status/{id}、/download/{id}、/materials/{id}/{file}
        │
        ▼
Celery + Redis：src/platform/celery_app.py
  │ 呼叫 main.run_pipeline(video_path, output_dir)
  ▼
影音前處理：src/preprocessing/preprocessor_optimized.py
  ├─ FFmpeg 轉 H.264/AAC MP4、抽 16 kHz 單聲道 WAV
  └─ 啟發式能量比估計 SNR；低於門檻時才做頻譜減法
        │
        ├──────────── 語音／文字軌 ───────────────┐
        │ Whisper 時間戳片段 → SemanticScorer      │
        │ s_text、teaching_stage、segment_summary │
        │ ProsodicAnalyzer：語速／RMS／F0／停頓   │
        │ s_prosodic + prosodic_available         │
        │                                          │
        └──────────── 視覺軌 ─────────────────────┤
          影片每段 25/50/75% 三個取樣點             │
          MediaPipe 食指 landmark → Kalman → GRU   │
          ROI → Tesseract OCR + OCR/SBERT 比對     │
          CLIP 影像／ASR 關鍵字 cosine 對齊         │
          視覺可靠度 R_visual（可用訊號加權 + EMA） │
                                                   ▼
融合：main.py
  ├─ 預設 Adaptive Fusion：β_eff=β×R_visual
  │  Score=(1−β_eff)×s_text + β_eff×s_visual
  ├─ 有效韻律分數才套用中心化倍率
  └─ USE_MLP_FUSION=True 時改走 12 維 MLP；目前預設 False
        ▼
摘要選段：semantic_sliding_window
  門檻／連續片段合併／單段長度限制／總摘要預算／依分數排序後選取
        ▼
FFmpeg 匯出：video_exporter_optimized.py
  summary.mp4 + index.json + timeline.json（原片與摘要輸出時間對照）
        ▼
Celery 結果 → FastAPI 狀態回應 → 前端時間軸、片段索引、統計與影片播放
```

圖中的語音/文字軌與視覺軌代表按片段整理的邏輯特徵分支，不代表兩個 worker 同時並行運算；目前 `run_pipeline()` 依序計算語意與韻律，再逐片段取樣處理影格。Web 層只把整部影片任務交給 Celery 背景 worker。

### 1.2 區塊責任與實際入口

| 區塊 | 實際程式 | 資料責任 |
|---|---|---|
| 前端 | [`src/platform/static/index.html`](../keytake-ai/src/platform/static/index.html) | 本機影片預覽、上傳、輪詢狀態、顯示索引與統計、切換原片／摘要時間軸 |
| API 匯入相容層 | [`src/platform/api.py`](../keytake-ai/src/platform/api.py) | 匯出優化版 FastAPI `app`，保留既有 Uvicorn 啟動路徑 |
| API 與檔案 I/O | [`src/platform/keytake_api_optimized.py`](../keytake-ai/src/platform/keytake_api_optimized.py) | 驗證、接收與儲存上傳、派送任務、回傳狀態、提供檔案下載 |
| 背景任務 | [`src/platform/celery_app.py`](../keytake-ai/src/platform/celery_app.py) | Celery 任務呼叫 `run_pipeline`、匯出摘要與索引；Redis 是 broker/backend |
| 主處理管線 | [`main.py`](../keytake-ai/main.py) | 串接前處理、ASR、語意、韻律、視覺、融合、選段和學習素材 |
| 特徵實驗入口 | [`tools/extract_features.py`](../keytake-ai/tools/extract_features.py)、[`src/fusion/feature_extractor.py`](../keytake-ai/src/fusion/feature_extractor.py) | 對影片抽取可重用的每片段 JSON 特徵；這是離線實驗路徑，不是 Web request 的獨立服務 |
| 實驗與評估 | [`tools/train_mlp_fusion.py`](../keytake-ai/tools/train_mlp_fusion.py)、[`tools/run_grid_search.py`](../keytake-ai/tools/run_grid_search.py)、[`tools/run_loocv.py`](../keytake-ai/tools/run_loocv.py)、[`tools/run_ablation_batch.py`](../keytake-ai/tools/run_ablation_batch.py) | 使用標註範圍正規化資料，訓練或比較融合候選 |

## 2. 從使用者操作到摘要影片：完整資料流程

以下步驟按執行先後排列。Web 服務流程由上傳觸發；離線 CLI 則可直接呼叫 `main.py`，不經 FastAPI、Redis 或 Celery。

| # | 目的、輸入與實際處理 | 輸出與下一步 | 程式位置 |
|---|---|---|---|
| 1 | 使用者在單頁前端選擇影片。前端先用 `URL.createObjectURL(file)` 顯示本機預覽；預覽並不代表檔案已送到伺服器。 | `FormData` 的 `file` 欄位送至 `/upload`。 | `index.html`：`handleFile()`、`uploadVideo()` |
| 2 | FastAPI 檢查檔名、副檔名是否在 mp4/mov/mkv/webm/avi/m4v 白名單，逐塊讀取上傳資料；超過 `KEYTAKE_MAX_UPLOAD_MB`（預設 500 MiB）或空檔會拒絕。 | 以 UUID 命名寫入 `tmp/uploads/`（可由 `KEYTAKE_UPLOAD_DIR` 改路徑），再交給 Celery；副檔名檢查不是影片內容的完整格式鑑定。 | `keytake_api_optimized.py`：`upload_video()` |
| 3 | 預設模式先探測 Redis broker 與 backend，再以相同 UUID 作 Celery task ID 排程。只有明確設定 `KEYTAKE_SYNC_FALLBACK=1` 才改由 API 程序背景執行緒處理；該模式的狀態只存程序記憶體。 | 前端取得 `task_id`，每 2 秒呼叫狀態 API。Celery 結果保存期限為 24 小時。 | `keytake_api_optimized.py`：`upload_video()`、`get_status()`；`celery_app.py` |
| 4 | 前處理以 FFmpeg 轉成固定 30 fps、H.264/AAC、yuv420p MP4，再抽成 16 kHz、單聲道、16-bit PCM WAV。 | 暫存影片與 WAV 路徑，以及 `snr`、`snr_type`、`denoised` metadata。 | `preprocessor_optimized.py`：`convert_to_mp4()`、`extract_audio()`、`preprocess()` |
| 5 | 以 25 ms 音框、10 ms hop 算每框平均功率；估計值為 `10 log10(全片平均功率 / 最低 10 百分位音框功率)`。低於 `AUDIO_SNR_THRESHOLD=10` 時，以最低 20% RMS 音框建立頻譜輪廓並做頻譜減法。 | 若降噪有執行，WAV 會被覆寫；同一 WAV 提供給 Whisper 與韻律分析。此 SNR 是啟發式能量比，不是有乾淨參考訊號的實測 SNR。 | `preprocessor_optimized.py`：`estimate_snr()`、`apply_spectral_subtraction()` |
| 6 | Whisper 接收 WAV。`WHISPER_MODE=local` 使用 `WHISPER_MODEL`（預設 `base`）；`remote` 才把音訊傳到 OpenAI transcription API，並要求 segment timestamps。 | 片段清單 `{start, end, text}`，時間單位為秒；metadata 記錄 requested/actual backend、模型與 fallback 原因。 | `transcriber_optimized.py`：`transcribe_with_metadata()`、`_transcribe_local()`、`_transcribe_remote()` |
| 7 | 轉錄結果剔除空文字、負起點、非正長度或不合格式片段，依起訖排序；只有文字相同且較短片段至少 80% 被時間重疊覆蓋時才合併。 | 時間對齊且去除高度重複的文字片段。這是重複片段合併，不是一般 ASR 幻覺偵測器。 | `transcriber_optimized.py`：`_deduplicate_segments()` |
| 8 | 語意評分依 `SEMANTIC_SCORER_MODE` 選擇路徑。SBERT 模式下，TF-IDF 取每段非零詞權重平均並以本批最大值縮放；SBERT 取片段嵌入對提示語料的最大 cosine，兩者依 `alpha_tfidf` 混合（預設 0.5）。 | 每段增加 `s_text`、`teaching_stage`、`stage_label`、`segment_summary`。SBERT/TF-IDF 不分類階段，會使用 `unclassified`。LLM 模式另可透過遠端服務產生分數、階段與摘要；失敗會有降級行為。 | `semantic_scorer_optimized.py`：`score()`、`tfidf_scores()`、`sbert_scores()`、`_llm_scores()` |
| 9 | 韻律分析用 librosa 讀取同一 WAV，依 Whisper 起訖裁切每段；段長至少 0.1 秒才計算。 | 對有效段輸出 `s_prosodic` 及 `prosodic_available=True`；無效段保留中性 0.5 並標 `False`，使下游知道不是觀測結果。 | `prosodic_analyzer_optimized.py`：`analyze_segments()` |
| 10 | 韻律原始量包含逐字稿非空白字元/秒、平均 RMS、pyin 基頻相對標準差、低於相對峰值 −40 dB 的 RMS frame 比例。每部影片有效片段內各自以 P10/P90 縮放，再加權；慢語速、較大音量、較大音高變化、較少停頓方向設為高分。 | 片段內相對韻律分數；不代表跨影片校準，也不是教學重要性標籤。 | `prosodic_analyzer_optimized.py`：`_estimate_speech_rate()`、`_compute_volume()`、`_compute_pitch_variation()`、`_compute_pause_ratio()`、`_normalize_features()`、`_compute_importance()` |
| 11 | 每個 ASR 片段在原片 25%、50%、75% 位置取三幀。MediaPipe 找手部 landmark；系統取第一隻手食指尖第 8 號點，Kalman 平滑後放入最近 15 點軌跡。 | 手部座標、移動變異、GRU/規則分類器結果。沒有手或模型／偵測器失敗時回傳 idle/零分路徑，不能因此推斷影片沒有教學內容。 | `main.py`：`run_pipeline()`；`hand_tracker.py`：`process_frame()` |
| 12 | GRU 輸入 20×4 的 `[x_norm,y_norm,dx_norm,dy_norm]` 序列，輸出 5 類手勢的 logits，再以 softmax 得類別機率；分數映射表決定是否達 0.7 ROI 觸發門檻。 | ROI 中心與手勢 intent score。它分類的是軌跡模式（指引／書寫／強調／過渡／idle），不是辨識「教師正在解釋哪個概念」。 | `gesture_classifier.py`：`extract_trajectory_features()`、`GestureClassifier.classify()`；`hand_tracker.py` |
| 13 | 對觸發 ROI 或整幀先做模糊檢查，低於 Laplacian 變異門檻時以 OpenCV bicubic 插值放大；Tesseract `image_to_data` 偵測文字框，必要時 `image_to_string` OCR。OCR 文字與 ASR 關鍵詞進 SBERT cosine，低分時混入字元命中；無 OCR 時用手框／文字框 IoU 作備援，並可加黃色 HSV 螢光筆比例。 | 每幀 `VisualScoreResult`；注意名稱 `SRGANEnhancer` 是包裝類，實際 `enhance_roi()` 是 bicubic resize，沒有執行已訓練 SRGAN。 | `visual_scorer.py`、`text_detector.py`、`srgan.py`、`sbert_calculator.py` |
| 14 | CLIP 將 ROI 與最多 20 個 jieba 關鍵字分別編碼到共同向量空間，計算 image/text embedding cosine，取關鍵字分數平均。若 CLIP 不可用或分數為 0，視覺合成會直接使用 OCR 路徑分數；否則依 `CLIP_WEIGHT` 混合，觸發手勢時再乘 intent score。 | 每幀視覺候選分數。程式沒有把 CLIP 原始分數寫入輸出快取，因此舊 cache 無法事後重做 CLIP 權重 sweep。 | `clip_scorer.py`：`compute_clip_score_with_keywords()`；`main.py`；`feature_extractor.py` |
| 15 | 可靠度將可用訊號轉換後按 0.35 手部穩定、0.25 偵測信心、0.25 文字密度、0.15 接近度加權，對存在的訊號重整權重，並用 EMA 平滑。每段取視覺分數最高的取樣幀及其可靠度。 | `visual_reliability` 與 `gesture_intent_score`。目前 `HandTracker.process_frame()` 沒有回傳 MediaPipe landmark confidence，因此建構訊號時該項通常缺席；權重會只在可用訊號中重整。 | `visual_reliability.py`：`estimate()`、`_compute_raw_reliability()`、`build_signals_from_tracker_result()`；`main.py` |
| 16 | Adaptive Fusion 依 α、β、R 組成分數；再依 `prosodic_available` 決定是否乘上韻律倍率。若 `USE_MLP_FUSION=True`，則走 checkpoint 的 12 維 MLP 輸出，韻律不再另套解析式倍率。 | 一個 `[0,1]` 融合分數/片段。當前設定 `USE_MLP_FUSION=False`，預設實際走解析式融合。 | `adaptive_fusion.py`：`fuse_scores()`；`main.py`：`_apply_prosodic_adjustment()`、`run_pipeline()` |
| 17 | 選段器先驗證片段時間及分數，閾值為 `max(FUSION_SCORE_THRESHOLD, median(scores))`（至少 5 段時）；連續高分片段合併，再拆分過長片段。候選依分數（或可選 score/second）排序，在 `duration × MAX_SUMMARY_RATIO` 預算內選取，最後回時間順序。 | 原片時間軸的選段清單。高分不保證入選：候選可能低於門檻、與相鄰段合併、被分數較高者占用預算，或因最大比例限制而排除。 | `adaptive_fusion.py`：`semantic_sliding_window()` |
| 18 | FFmpeg 逐段剪出含 0.3 秒前後 padding 的片段，重疊 padding 區間合併，最後 concat 串接；probe 實際時長並建立 source/output mapping。 | `summary.mp4`、`timeline.json`、`index.json`。索引時間仍是原片時間；timeline 對照把原片區間映射到摘要影片輸出時間。 | `video_exporter_optimized.py`：`export_summary_video()`、`_normalize_segments()`、`_build_timeline_mapping()`、`export_index()` |
| 19 | Celery 回傳片段陣列、課程摘要、原片/摘要時長、節省比例與 timeline。前端輪詢 `/status/{task_id}`，用 timeline 畫摘要時間軸與跳轉；缺對照資料時會停用摘要切換。 | 使用者可播放/下載摘要、看片段索引與統計、匯出索引 JSON。原片播放來自瀏覽器最初建立的本機 Blob URL，不由 API 提供原片串流。 | `celery_app.py`、`keytake_api_optimized.py`、`index.html`：`pollStatus()`、`renderResult()`、`getSummaryRange()`、`seekToSegment()`、`renderTimeline()` |

`StudyMaterialsGenerator.export_all()` 會在主 pipeline 產生 `study_notes.md`、`mind_map.json`、`flashcards.json`、`chapters.json`。這些是可透過 materials API 讀取的伺服器檔案；目前 Celery 的狀態回應及單頁 UI 沒有把四種教材直接包成結果卡片呈現。

### 2.1 假設教學影片走一次完整流程

以下數字是**教學用假設值**，不是實驗紀錄。假設使用者上傳一部 240 秒、講解二次方程式的影片：

1. 前端先在瀏覽器預覽影片，接著把檔案送到 `/upload`。API 回傳 UUID task ID；Celery worker 呼叫 `run_pipeline()`，而不是由前端直接呼叫 AI 模型。
2. FFmpeg 產生 MP4/WAV。假設 SNR heuristic 結果為 24 dB，因大於 10 dB，程式不啟動頻譜減法；這不表示錄音一定清晰，只代表估計值未低於門檻。
3. Whisper 回傳假設片段 `start=52.3, end=60.4, text="先把方程式整理成標準形式，再利用公式求根"`。這個 8.1 秒區間成為模態對齊 key。
4. SBERT/TF-IDF 產生假設 `s_text=0.82`，韻律分析產生 `s_prosodic=0.70` 且 availability 為 true。三個取樣影格中，假設最高視覺分 `s_visual=0.62`、可靠度 `R_visual=0.50`。
5. 使用 `ALPHA=0.60,BETA=0.40`，Adaptive Fusion 得到 `β_eff=.40×.50=.20`、`S=.80×.82+.20×.62=.78`。韻律倍率為 `1+.15×(.70-.50)×2=1.06`，所以微調後為 `.78×1.06=.8268`。
6. 若同批至少 5 個片段的 median score 是 0.65，且 0.8268 高於門檻，該片段會成為候選；但 240 秒影片的總預算為 144 秒，若前面較高優先度片段已用完預算，它仍可能不入選。
7. 若入選，匯出器以約 `[52.0,60.7]` 秒（前後各約 0.3 秒 padding）剪輯。若前面沒有其他摘要片段，該片段內原始 `[52.3,60.4]` 秒約映射到摘要影片 `[0.3,8.4]` 秒；若前面已有摘要片段，output offset 還會加上前面片段的實際編碼時長。
8. 前端取得結果後仍以原片時間顯示索引；摘要模式則透過 timeline mapping 尋找摘要影片時間。以上示範不代表此句子實際被選中，也不表示分數等於人工標註機率。

## 3. Web API 與前端資料契約

| HTTP | 功能 | 主要回應／限制 |
|---|---|---|
| `GET /` | 回傳靜態單頁前端 | `index.html`，設定禁止快取 |
| `GET /health` | 健康檢查 | `{"status":"ok"}`；不表示 Redis/worker 一定正常 |
| `POST /upload` | 上傳並排程 | 202；`task_id`、`status`、`mode`。副檔名白名單、500 MiB 預設上限 |
| `GET /status/{task_id}` | 取得狀態／結果 | queued/processing/done/error；任務 ID 必須符合 UUID 格式 |
| `GET /download/{task_id}` | 下載摘要 | `summary.mp4`；檔案尚未生成回 404 |
| `GET /materials/{task_id}/{filename}` | 讀取學習素材 | 僅允許 `study_notes.md`、`mind_map.json`、`flashcards.json`、`chapters.json` |

預設是 FastAPI + Celery + Redis；同步執行緒 fallback 是單機開發用途，使用程序內 dict 保存狀態，程序重啟後不持久。程式碼中未見使用者登入、權限控管、任務取消端點或多租戶隔離；部署到公開網路前需另行補上安全與資源控管。

`run_pipeline()` 也會嘗試用 [`study_materials_generator_optimized.py`](../keytake-ai/src/output/study_materials_generator_optimized.py) 產生章節、Markdown 筆記、心智圖及閃卡；此生成失敗會記錄錯誤但不使摘要主流程失敗。Web 的 materials API 可讀取已輸出的檔案，但目前單頁前端的主要結果 UI 顯示課程摘要、關鍵概念、片段索引、時間軸與時長統計，沒有完整呈現筆記/心智圖/閃卡的互動面板。

## 4. 核心技術實作

### A. Whisper 語音辨識

**用途與資料**：將前處理後 WAV 轉成有時間戳的語音片段，資料格式為 `{"start": 秒, "end": 秒, "text": 字串}`。時間戳是之後將文字分數、聲音韻律、畫面取樣與摘要剪輯對齊的共同座標。

**模型與模式**：[`config.py`](../keytake-ai/config.py) 設 `WHISPER_MODEL="base"`；[`transcriber_optimized.py`](../keytake-ai/src/semantic/transcriber_optimized.py) 在 local 模式呼叫 `whisper.load_model(model).transcribe(..., word_timestamps=False)`，程序內以 model 名稱快取模型。remote 模式使用 OpenAI transcription API 的 `verbose_json` 與 segment timestamps。

**錯誤處理**：local 模式缺 Whisper 套件或模型時會報錯；remote 缺 OpenAI 套件／金鑰或呼叫失敗時會記錄 fallback reason，改試本機模型。文字重複處理只合併同文字、高度時間重疊片段。程式沒有一般性 hallucination detector；因此 README 舊圖中的「幻覺過濾」不精確。

**影響**：漏字、錯字、時間漂移會影響 `s_text`、jieba 關鍵字、OCR/CLIP 的文字查詢、韻律 speech-rate 分子及摘要索引。摘要端無法自動修復錯誤轉錄。

### B. 文字特徵：TF-IDF、SBERT 與 LLM 的差異

- **TF-IDF**：`TfidfVectorizer(max_features=TFIDF_TOP_K)` 對本次影片各逐字稿片段建立詞權重。分數是片段內非零 TF-IDF 平均，再以本片最大值縮放。它表示詞在本批文件中的相對辨識度，不是句子語意理解。
- **SBERT**：`paraphrase-multilingual-MiniLM-L12-v2` 將文字片段與 `PROMPT_CORPUS` 提示句轉成向量，L2 normalize 後算 cosine，對每片段取與語料的最大相似度。SBERT 模式最終 `s_text = 0.5×TFIDF + 0.5×SBERT`（除非呼叫端傳入不同 `alpha_tfidf`）。
- **LLM**：只在 `SEMANTIC_SCORER_MODE=llm` 時，對每段文字要求 JSON score/stage/summary；選擇 OpenAI 或 Groq 會把逐字稿送至第三方，Ollama 則呼叫本機服務。錯誤值有 fallback，但這不是可重現的人工標註模型。

本專案不是把「任何句子和任何句子」全部兩兩比較；主語意分數是片段對提示語料的最大相似度。視覺子流程的 `SBERTCalculator` 則另外比較 OCR 文字與 ASR 關鍵字字串。二者功能不同。

### C. CLIP 視覺－文字對齊

CLIP 的 image encoder 和 text encoder 把影像與文字投影到同一嵌入空間。實際程式用 `CLIPModel`、`CLIPProcessor` 載入 `openai/clip-vit-base-patch32`，在 CPU/GPU 選擇可用裝置、`eval()` 後以 `torch.no_grad()` 推論；`CLIPScorer` 是 singleton，避免重複載入。

\[
\operatorname{cos}(v_i,v_t)=\frac{v_i\cdot v_t}{\|v_i\|_2\|v_t\|_2}
\]

其中 \(v_i\) 是 ROI 影像 embedding，\(v_t\) 是 ASR 關鍵字文字 embedding。程式將 cosine 裁至 `[0,1]`，並對最多 20 個關鍵字取平均。這個 cosine 不是校準後的「教學重要性機率」，也不保證值越大越能提升人工評估。

ROI 是 Region of Interest，即由手部中心裁切出的局部影像；沒有觸發手勢時使用整幀作影像輸入。視覺混合為 `s_visual=(1−CLIP_WEIGHT)×OCR視覺分數+CLIP_WEIGHT×CLIP分數`，只有 CLIP 分數大於 0 才走此混合；否則直接沿用 OCR 視覺分數。`CLIP_SCORE_THRESHOLD` 目前只有 getter，程式未以它截斷或過濾分數。特徵 JSON 沒有保留原始 CLIP 分數，因此無法只讀舊快取重新調權。

### D. MediaPipe 手勢特徵

MediaPipe Hand Landmarker 偵測手部骨架；程式取第一隻手 index tip landmark 8，將 normalized x/y 乘影格寬高成像素座標。OpenCV Kalman filter 平滑位置，deque 保存最多 15 點。`gesture_classifier.py` 將座標轉成 `[x_norm,y_norm,dx_norm,dy_norm]`，補齊或截斷到 20 個取樣點，再交給雙層 GRU。沒有權重或 PyTorch 時退回規則分類器。

規則式後備分類器另計算 x/y 位移變異數、相鄰 x 位移方向反轉次數、相鄰點歐氏距離總和（path length）及首末點位移（displacement），用門檻判斷五類；這些值不是 GRU 的額外輸入。系統沒有明確計算手勢「停留時間」特徵；現有 GRU 使用取樣點序列而非帶實際秒數的時間間隔，因此不能聲稱模型已學到 dwell time。

分類標籤與設定的 intent score：

| 類別 | score |
|---|---:|
| POINTING | 1.00 |
| WRITING | 0.85 |
| EMPHASIS | 0.90 |
| TRANSITION | 0.20 |
| IDLE | 0.10 |

intent score 達 0.7 才觸發手部中心 ROI；觸發時視覺分數還會乘 intent score。這只表示軌跡分類器偏向哪些已定義類別，不是模型理解了講師的語意、手指指向內容或實際教學意圖。沒有手時清空座標歷史，回 idle 資料；下游沒有有效視覺證據時可靠度為 0。

### E. SNR 與音訊前處理

理想的功率訊號雜訊比定義是：

\[
\mathrm{SNR}_{dB}=10\log_{10}\frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}}
\]

程式沒有 clean reference track，而是把全片平均 frame power 當訊號功率、最低 10 百分位 frame power 當噪音底估計。假設最低能量音框大致代表背景噪音；若整段皆有語音、靜音比例不同或背景聲不穩，估值就可能偏差。低於 10 dB 時才用最低 20% RMS frame 的 magnitude median 做頻譜減法。該門檻是目前程式設定，不是以代表性資料與人工音質標籤驗證的最佳門檻。

目前三部訓練影片的啟發式估值均高於 20 dB，因此 5/10/15/20 dB 探索門檻在這三片都不會觸發降噪。樣本太少且沒有音質標註，不能據此宣稱 10 dB 已有效或調低／調高門檻。處理失敗會印出訊息並保留擷取音訊；這使主流程能繼續，但失敗資訊應在實驗記錄中保留。

### F. 韻律特徵

| 量 | 程式算法 | 解讀及限制 |
|---|---|---|
| speech_rate | Whisper 文字非空白字元數 / 片段秒數 | 不是音節、詞或實際偵測發音速率；轉錄長短會影響估算 |
| volume | `librosa.feature.rms` 平均 | 受錄音距離、增益、麥克風影響 |
| pitch_variation | `librosa.pyin` 有聲 F0 的 `std/mean` | 不是音高均值，也可能受 pyin 錯誤影響 |
| pause_ratio | RMS 轉相對峰值 dB，低於 `silence_threshold_db=-40` 比例 | 是能量低比例的啟發式，不是語言學停頓辨識 |

四項各自在本片有效段落以 P10/P90 min-max clip 到 `[0,1]`。最後把慢語速 `1−normalized_rate`、音量、音高變化、少停頓 `1−normalized_pause` 依目前設定權重 0.30、0.30、0.25、0.15 加總；權重會 normalize 至總和 1。無 librosa、音訊錯誤、片段過短則輸出 0.5 佔位並 `prosodic_available=False`。這些量描述表達方式，不是教學價值的標準答案。

### G. Adaptive Fusion

公式中 \(s_t\) 是文字分數、\(s_v\) 是視覺分數、\(R_v\) 是視覺可靠度、\(\alpha,\beta\) 是基礎模態權重且 \(\alpha+\beta=1\)：

\[
\beta_{\mathrm{eff}}=\beta R_v,\quad
\alpha_{\mathrm{eff}}=1-\beta_{\mathrm{eff}},\quad
S=\alpha_{\mathrm{eff}}s_t+\beta_{\mathrm{eff}}s_v
\]

例如 \(s_t=0.82,s_v=0.62,\beta=0.4,R_v=0.5\)：\(\beta_{\mathrm{eff}}=0.20\)，\(\alpha_{\mathrm{eff}}=0.80\)，因此 \(S=0.80×0.82+0.20×0.62=0.78\)。視覺可靠度下降時，視覺模態影響自動下降。

可靠度原始四個 component 是手部穩定度 `exp(-variance/HAND_VARIANCE_THRESHOLD)`、偵測 confidence、文字框 union 面積比 `r` 的 `min(1, log1p(20r)/log1p(20))`、食指至最近文字框中心距離除以畫面對角線後的 `exp(-4d)`。原始 component 權重為 0.35、0.25、0.25、0.15，對可用 component 重新正規化。非零可靠度用 `ema_alpha=0.3` 的 EMA 平滑，非零結果下限為 0.05；若無視覺分數或沒有可用手/文字訊號則重置 EMA 並回 0。**目前 HandTracker 不提供 confidence 欄位**，所以實際融合通常只使用其餘可用項，需避免把四項都描述成每幀一定有值。

目前程式設定為 `ALPHA=0.6,BETA=0.4`；三部訓練影片搜尋觀察到候選 0.5/0.5、macro F1=0.5341，但屬單人標註小樣本探索，未更新為正式預設。固定 0.6/0.4 也不是經充分外部驗證的最佳值。靜態融合等同把視覺可靠度設 1；Adaptive 在視覺證據差時抑制視覺輸入。

核心實作可簡化讀成：

```python
beta_eff = beta * reliability
alpha_eff = 1.0 - beta_eff
score = alpha_eff * s_text + beta_eff * s_visual
```

`fuse_scores()` 會先檢查輸入分數與權重都是有限數值、範圍在 `[0,1]`，並要求 `alpha + beta == 1`；`reliability` 夾在 `[0,1]` 後才計算。因 `alpha_eff` 是補足權重，低可靠度不會把總權重縮小，而會把被移出的權重交回文字模態。

### H. MLP Fusion（實驗性、未啟用）

MLP 是前饋全連接網路。`MLP_FEATURE_NAMES` 目前 12 維：

| 維度 | 特徵 |
|---:|---|
| 1 | `s_text`：文字相對重要分數 |
| 2 | `s_visual`：OCR/CLIP/手勢調整後視覺分數 |
| 3 | `visual_reliability`：視覺可靠度 |
| 4 | `gesture_intent_score`：手勢類別映射值 |
| 5 | `s_prosodic`：韻律啟發式分數 |
| 6 | `prosodic_available`：是否有有效韻律量測的 0/1 mask |
| 7–12 | definition、derivation、example、summary、transition、qa 六類 one-hot；unclassified 為全 0 |

網路為 `12→32→16→1`，兩個隱藏層之間使用 ReLU，含 Dropout 0.3，輸出 sigmoid。訓練用 BCELoss + Adam（預設 100 epochs、lr 0.001），每 20 epoch 記錄訓練／驗證 loss；驗證 loss 最佳時存 checkpoint。正標籤預設以 `overlap / GT duration >= 0.5` 定義，與 evaluator 的 recall hit 門檻一致。

程式定義的特徵順序核心如下（以實際 `MLP_FEATURE_NAMES` 為準）：

```python
MLP_FEATURE_NAMES = (
    "s_text", "s_visual", "visual_reliability",
    "gesture_intent_score", "s_prosodic", "prosodic_available",
    *(f"teaching_stage:{stage}" for stage in TEACHING_STAGES),
)
```

前 6 維保留模態分數和韻律缺值遮罩，後 6 維是 teaching stage one-hot；checkpoint 也保存 feature version 與名字順序，避免把舊 11 維或欄位順序不同的權重誤載入。

`train_mlp_fusion.py` 要求明確提供獨立測試 feature/annotation 配對；測試影片不進 train 或 group validation。每部影片先依 `eval_range_sec` 裁切時間，訓練片段按 video ID 分組切分，避免同一影片片段落入 train/validation 兩側。checkpoint 保存 `input_dim`、`feature_version`、特徵次序、階段與 label basis；載入不相容會錯誤並由融合包裝函式記錄後退回解析式公式。

本次新模型 `weights/mlp_fusion_12d.pth` 以 3 部訓練影片的 240 個片段、118 個 GT 區間訓練/驗證，在 `test_10min` 執行了一次 12 維 MLP 評估，結果 Recall=0.462、Precision=0.545、F1=0.500。三部訓練影片中一部作 group validation，其 validation loss=0.6693、best epoch=74。這是小樣本結果，不足以宣稱泛化或效能提升。需同時揭露：根目錄 README 保留了較早以 `test_10min` 掃描 β 的歷史表格，因此不能把此影片稱為完全 untouched 的 holdout，也不能再拿它選參；新 MLP 的單次評估不會消除過往的資料使用紀錄。`USE_MLP_FUSION=False`；CLI 新模型路徑是 `weights/mlp_fusion_12d.pth`，checkpoint 檔案本身屬本機產物，應按實驗需求另行管理，本文不把權重當程式碼提交。

### I. 摘要選段、輸出時間與「分數不等於入選」

`semantic_sliding_window()` 接受片段與同長度分數清單，先驗證 finite/time range。片段依時間排序；至少 5 段時閾值取 `max(FUSION_SCORE_THRESHOLD, median(scores))`，否則用底線。相鄰的高分段在符合最大單段長度時合併，重疊片段合併，過長片段再拆分。

總預算為：

\[
B=T_{\mathrm{original}}\times\mathrm{MAX\_SUMMARY\_RATIO}
\]

預設 `MAX_SUMMARY_RATIO=0.6`，所以摘要預算最多為指定原片長的 60%。單段上限實際為 `min(max(min(SLIDING_WINDOW_MIN_SEC, budget)×2, MIN_SUMMARY_SEC), budget)`；預設 min sec=10、min summary=2 時，單段上限通常為 20 秒（若總預算更短則受預算限制）。程式沒有強制每個片段都至少 2 秒，也沒有保證摘要總長至少 2 秒；`MIN_SUMMARY_SEC` 在這裡是計算單段上限的參數。候選再按總分排序，逐個放進預算，最後按原片時間排序。若無片段越過門檻，會嘗試保留最高分片段；仍會受預算限制。

所以「模型分數高」不等於「最後一定入選」。門檻、候選合併、段長限制、總預算競爭均會影響結果。

選段器在門檻候選產生與合併後，實際以分數排名並填入預算；下列為 `semantic_sliding_window()` 中的核心片段：

```python
ranked = sorted(candidates, key=_priority, reverse=True)
kept = []
used = 0.0
for segment in ranked:
    length = segment["end"] - segment["start"]
    if used + length <= budget + 1e-9:
        kept.append(segment)
        used += length
return sorted(kept, key=lambda item: (item["start"], item["end"]))
```

`_priority` 預設使用片段總分；只有明確選擇 `score_per_second` 時才用分數/時長。實際程式會逐候選累加時長，超出 budget 的候選跳過；最後依來源時間排序，保持影片的時間脈絡。

匯出器依原片 start/end 排序，加入前後 0.3 秒 padding；有重疊的 padding clip 合併。每段用 FFmpeg 重新編碼 H.264/AAC，再用 concat demuxer 串接。`timeline.json` 保存 `source_start/source_end` 與 `output_start/output_end`，其中 output 時長以 probe 後比例校準，供前端摘要模式正確跳轉。影片重編碼可能造成少量邊界或時間基準誤差，mapping 是依實際 probe 長度估算，不是逐影格精密追蹤。

## 5. 設定、依賴與資料安全

### 5.1 主要設定

| 設定 | 目前程式值 | 正確解讀 |
|---|---:|---|
| `WHISPER_MODE` | `.env.example` 為 `local` | local 用 Whisper `base`；remote 會把音訊傳 OpenAI |
| `SEMANTIC_SCORER_MODE` | 範本 `sbert`，程式預設 `sbert` | `llm` 才走外部 LLM；設定檔可能覆蓋範本 |
| `AUDIO_SNR_THRESHOLD` | 10 dB | 啟發式降噪觸發門檻，尚未充分驗證 |
| `ALPHA/BETA` | 0.6/0.4 | 目前程式基準，不是目前資料搜尋出的最佳候選 |
| `CLIP_WEIGHT` | 0.3 | 現行程式混合係數；無原始 CLIP cache，尚未以同一驗證流程校準 |
| `PROSODIC_WEIGHT` | 0.15 | 現行程式韻律倍率係數，未改成 sweep 候選 |
| `USE_MLP_FUSION` | `False` | Web/CLI 主流程仍走解析式 Adaptive Fusion |
| `MLP_FUSION_MODEL_PATH` | `weights/mlp_fusion_12d.pth` | 只在明確啟用 MLP 時使用；模型權重不是程式碼依賴的一部分 |

### 5.2 套件與外部服務

Python 套件列於 [`requirements.txt`](../keytake-ai/requirements.txt)：Whisper、PyTorch、sentence-transformers、scikit-learn、OpenCV、MediaPipe、Tesseract wrapper、librosa、Transformers/CLIP、FastAPI、Celery、Redis client 等。另需系統安裝 FFmpeg 與 Tesseract executable；Web 非同步模式還需 Redis server 與 Celery worker。

預設使用本機 Whisper 與 SBERT，但首次執行可能需從 Hugging Face/模型來源下載權重；「本機推論」不代表首次安裝完全無網路。遠端 OpenAI/Groq LLM 與 Whisper API 會傳送輸入資料，必須明確選用並按資料政策授權。LLM 語意 scorer 在初始化失敗時有降級路徑；學習素材生成器也可能按設定選遠端 LLM 或規則式生成。

根目錄 `.gitignore` 忽略 `.env`、影音檔與常見模型快取；實際提交仍須再檢查 staged diff，不能只依賴 ignore 規則。上傳 API 目前沒有登入驗證，適合受控開發環境，不應直接視為公開部署安全設計。

## 6. 實際實驗狀態與不可混淆的結果

1. 四部影片有 12 維 `keytake-multimodal-12d-v3` 特徵；三部用於訓練／驗證，一部 `test_10min` 提供 MLP 評估，但有早期 β 掃描紀錄，不是從未使用過的獨立 holdout。
2. `test_10min` GT 只涵蓋 `[0,230.08]` 秒；雖然 feature JSON 延伸至約 600 秒，訓練／測試工具現在都會先用 `normalize_evaluation_data()` 裁切並歸零時間。
3. 三部訓練影片的 α/β search 得到 0.50/0.50、macro F1=0.5341，完整結果見 [`grid_search_12d_training_only.json`](../keytake-ai/results/grid_search_12d_training_only.json)。`config.py` 仍為 0.6/0.4，因小樣本搜尋不作正式採用。
4. [`prosodic_weight_sweep_12d_training_only.json`](../keytake-ai/results/prosodic_weight_sweep_12d_training_only.json) 記錄的是**韻律權重** sweep：0.30 的觀察 macro F1=0.5611。它不是 CLIP weight sweep；目前 `CLIP_WEIGHT=0.3` 是程式既有設定，不可把上述 F1 歸因於 CLIP。
5. CLIP 原始分數未寫入特徵 cache，不能用目前 cache 重做 CLIP 權重比較。
6. 三部訓練影片的音訊啟發式 SNR 約為 23.23、36.89、27.45 dB（均高於 20 dB）。10 dB 降噪條件尚未以足夠多樣音訊和標註驗證。
7. 12 維 MLP 在 `test_10min` 新跑一次結果為 Recall=0.462、Precision=0.545、F1=0.500。根目錄 README 另保留有較早的 `test_10min` β 掃描歷史紀錄，因此不得稱此影片從未用於探索，更不得再以它選參。
8. Ground Truth 是單人標註，Cohen’s Kappa 為 null；結果只能描述目前樣本，不能稱為已證明的效能提升、正式最佳模型或具統計代表性的結論。

## 7. README／簡報中需以程式碼為準的差異

| 舊說法或容易誤讀處 | 實際程式 |
|---|---|
| Whisper 「幻覺過濾」 | 實作的是格式檢查及相同文字高度重疊片段合併；不是一般 ASR hallucination detector |
| SRGAN 增強 | `srgan.py` 使用 OpenCV bicubic resize，沒有載入 SRGAN/Real-ESRGAN 權重 |
| 四個可靠度訊號每次均可用 | `HandTracker` 沒有輸出 confidence；估計器只重整可用訊號權重 |
| Clip threshold 會篩選分數 | `CLIP_SCORE_THRESHOLD` 目前只有 getter，呼叫路徑未使用此門檻 |
| 手勢分類代表教師意圖理解 | 僅根據食指軌跡分類五種預定義動作並用分數觸發 ROI |
| α/β 是目前 Grid Search 最佳值 | 程式基準仍是 0.6/0.4；三部影片候選 0.5/0.5 未採用 |
| macro F1 0.5611 是 CLIP sweep | 記錄的 sweep 變數是 `prosodic_weight`，屬韻律權重探索 |
| `MIN_SUMMARY_SEC` 保證摘要至少這麼長 | 它參與最大單段長度計算，不是摘要總長度下限 |

以上差異應在投影片與論文答辯中主動澄清，而不是以模組名稱推導功能。
