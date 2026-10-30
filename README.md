# A Hybrid Semantic-Visual Fusion System for Automated Educational Video Summarization

本專案實作了一套教育影片自動摘要系統（KeyTake AI），結合語意分析與視覺特徵，
自動從教學影片中擷取重點片段，並輸出濃縮影片與結構化學習素材。

本研究成果已投稿至 **TANET 2026（臺灣網際網路研討會）**，第一作者。

---

## 研究動機

教學影片普遍偏長，學生複習時需反覆快轉定位重點，效率低落。
既有的自動摘要方法多依賴單一模態：純語意方法容易漏失板書等視覺資訊，
純視覺方法則對手勢雜訊敏感，在教師走動或手部離開板書時產生大量誤報。

本研究的出發點是：**教學場景的語意與視覺資訊天然互補**。
重要概念往往同時伴隨口語講解與板書書寫，但兩者的可靠度隨場景變動，
因此需要一個能動態調整各模態權重的融合機制。

---

## 系統架構

Web 入口為 FastAPI，預設透過 Celery/Redis 背景執行主處理管線。語意與視覺特徵依片段對齊後融合，再由時間預算選出摘要區段。完整逐步流程、公式與實際函式請見[系統架構與技術實作說明書](docs/KeyTake%20AI%20系統架構與技術實作說明書.md)。

```
使用者 → index.html → FastAPI /upload
                         │
                 Celery + Redis（預設）
                         │
           FFmpeg 轉檔、抽音訊、啟發式 SNR
               ┌─────────┴──────────┐
               │                    │
      Whisper → 文字/韻律       影片取樣幀
      TF-IDF/SBERT/可選 LLM     MediaPipe → GRU/ROI
                               OCR/SBERT + CLIP
               └─────────┬──────────┘
                  可靠度與特徵融合
               Adaptive（預設）/ MLP（停用）
                         │
          門檻、合併、單段限制、總時長預算
                         │
      FFmpeg 摘要 + index.json + timeline.json
                         │
             前端輪詢、索引、播放與下載
```

### 核心模組

| 模組 | 位置 | 說明 |
|------|------|------|
| 主處理管線 | `keytake-ai/main.py` | `run_pipeline()` 串接前處理、ASR、多模態分析、融合、選段與輸出 |
| 影音前處理 | `keytake-ai/src/preprocessing/preprocessor_optimized.py` | FFmpeg 格式轉換、抽音訊、能量比估計與條件式頻譜減法 |
| 語音與語意 | `keytake-ai/src/semantic/transcriber_optimized.py`、`semantic_scorer_optimized.py` | Whisper local/remote、TF-IDF/SBERT/可選 LLM；重複片段合併不是一般幻覺過濾 |
| 韻律分析 | `keytake-ai/src/semantic/prosodic_analyzer_optimized.py` | librosa 估算語速、RMS、F0 變化與低能量比例 |
| 手部與手勢 | `keytake-ai/src/visual/hand_tracker.py`、`gesture_classifier.py` | MediaPipe 食指軌跡、Kalman 平滑、GRU/規則式五類分類 |
| 視覺評分 | `keytake-ai/src/visual/visual_scorer.py`、`clip_scorer.py` | OCR/文字比對與 CLIP 跨模態相似度；`srgan.py` 實際使用 bicubic resize |
| 視覺可靠度 | `keytake-ai/src/visual/visual_reliability.py` | 對可用訊號重新加權並 EMA 平滑；目前 tracker 未提供 landmark confidence |
| 融合與選段 | `keytake-ai/src/fusion/adaptive_fusion.py`、`keytake-ai/src/fusion/mlp_fusion.py` | Adaptive 解析式融合為預設；12 維 MLP 需明確啟用，目前停用 |
| 摘要輸出 | `keytake-ai/src/output/video_exporter_optimized.py` | FFmpeg 剪輯、串接、索引與 source/output timeline mapping |
| API 與背景任務 | `keytake-ai/src/platform/keytake_api_optimized.py`、`celery_app.py` | FastAPI 上傳/狀態/下載，預設 Celery + Redis |

---

## 手勢意圖分類

GRU 分類器將食指軌跡序列分為五類，各類映射至視覺流程使用的 intent score。這些標籤描述預先定義的軌跡類別，不代表系統理解了講師教學意圖：

| 手勢意圖 | 說明 | 重要性分數 |
|---------|------|-----------|
| 指引 (Pointing) | 靜止指向某位置 | 1.00 |
| 強調 (Emphasis) | 局部反覆來回 | 0.90 |
| 書寫 (Writing) | 手腕小幅規律移動 | 0.85 |
| 過渡 (Transition) | 大幅移動換區 | 0.20 |
| 無手部 (Idle) | 未偵測到手 | 0.10 |

GRU 架構：雙層，hidden size 64，輸入維度 4（食指 x/y + 移動量），序列長 20 幀，
模型不存在時可降級為規則型分類器。視覺可靠度的 confidence component 目前未由 `HandTracker` 輸出，因此不是每次都可用。

---

## 自適應置信度加權融合

視覺可靠度 R_visual ∈ [0,1] 由四項可觀測訊號加權估計：

| 訊號 | 權重 | 說明 |
|------|------|------|
| 手部穩定度 | 0.35 | 食指位移變異數越低越穩定 |
| 偵測信心度 | 0.25 | 若 tracker 有提供則納入；目前 `HandTracker` 未傳出此欄位，因此通常缺席並重整其餘可用權重 |
| 板書文字密度 | 0.25 | 文字框面積佔畫面比例（對數壓縮） |
| 手板接近度 | 0.15 | 食指中心點與最近文字框的距離 |

加上指數移動平均（EMA）平滑，避免單幀跳動影響穩定性。可靠度計算會忽略缺少的訊號並重整權重；若沒有可用手部或文字訊號，回傳 0。

融合公式：

```
β_eff  = β × R_visual
α_eff  = 1 − β_eff
Score  = α_eff × s_text + β_eff × s_visual
```

當 R_visual = 1 時退化為固定加權；視覺不可靠時視覺權重自動趨近 0。

---

## 先前基準實驗（歷史結果，非目前 12 維 MLP 評估）

以下結果來自較早的融合評估腳本及其資料設定，原數值照錄以保留實驗紀錄。這些表格與近期 12 維 MLP 特徵實驗不是同一組評估，且只有四部影片、單人標註，不能作為泛化或效能提升已獲證明的結論。特別是下方曾以 `test_10min` 掃描 β；該資料因此不能再被描述為完全未觸碰的 held-out set，也不應再用於選參。

### 消融實驗（四部真實教學影片平均，α=0.6, β=0.4）

| 融合策略 | Recall | Precision | F1 | FAR | TSR |
|---------|--------|-----------|----|----|-----|
| 純語意 | 0.476 | 0.735 | 0.571 | 0.125 | 0.436 |
| **靜態融合** | **0.547** | **0.780** | **0.632** | **0.100** | 0.395 |
| 自適應融合 | 0.469 | 0.722 | 0.559 | 0.113 | 0.430 |

各項標準差：Recall ≤ 0.069、Precision ≤ 0.183、F1 ≤ 0.078、FAR ≤ 0.044、TSR ≤ 0.125。

在這組歷史樣本上的數值差異為：靜態融合 F1 +0.061、Recall +0.071、FAR −0.025。這是描述性結果，並非統計顯著或已驗證的效能提升。

### 逐部影片明細（Recall / F1 / FAR）

| 影片 | 純語意 | 靜態融合 | 自適應融合 |
|------|--------|----------|-----------|
| lecture_01（分數/小數） | 0.48 / 0.59 / 0.14 | 0.61 / 0.73 / 0.06 | 0.48 / 0.61 / 0.09 |
| lecture_02（三角學） | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 |
| lecture_03（心算） | 0.46 / 0.60 / 0.10 | 0.46 / 0.58 / 0.12 | 0.43 / 0.55 / 0.13 |
| test_10min（通用） | 0.46 / 0.46 / 0.19 | 0.62 / 0.57 / 0.15 | 0.46 / 0.44 / 0.17 |

### 歷史探索性參數敏感度（test_10min，曾掃描視覺權重 β；不作選參依據）

| β | Recall | Precision | F1 | FAR |
|---|--------|-----------|----|----|
| 0.0（純語意） | 0.462 | 0.455 | 0.458 | 0.189 |
| 0.1 | 0.462 | 0.500 | **0.480** | 0.147 |
| 0.4 | 0.462 | 0.417 | 0.438 | 0.169 |
| 1.0（純視覺） | 0.231 | 0.333 | 0.273 | 0.147 |

本組影片 R_visual 中位數約 0.05，視覺訊號整體偏弱。此結果至多描述這批影片的行為；因資料規模小且 β 掃描使用 `test_10min`，不能用來證明自適應機制有效，也不得再據此調整參數。

### 實驗設定說明

- 評估影片：四部公開教學影片，包含三部數學課程與一部 `test_10min`；各實驗依對應 annotation/`eval_range_sec` 定義範圍，不概稱四部皆取相同片長
- Ground Truth：由標註者依 Likert 1~5 分標註每段重要性，≥ 4 分列入 GT
- 本版為**單人標註可行性驗證（single-annotator pilot）**，Cohen's Kappa 列為後續工作
- 片段與 GT 重疊超過 50% 視為命中
- 近期 12 維 MLP 對 `test_10min` 的評估範圍為 `[0, 230.08]` 秒；此影片另有較早 β 掃描紀錄，不能視為全程 untouched 的測試集
- 近期另有四部影片的 12 維特徵實驗；三部訓練影片的 α/β 探索候選為 0.50/0.50（macro F1=0.5341），韻律權重探索候選 0.30（macro F1=0.5611）。後者是 **prosodic weight sweep，不是 CLIP sweep**；兩項均為小樣本探索，不作正式最佳設定。可查閱 [α/β 搜尋紀錄](keytake-ai/results/grid_search_12d_training_only.json)與[韻律權重 sweep 紀錄](keytake-ai/results/prosodic_weight_sweep_12d_training_only.json)。
- 12 維 MLP 在 `test_10min` 的一次新評估為 Recall=0.462、Precision=0.545、F1=0.500。因該影片曾出現在上方歷史 β 掃描，不能把它宣稱為完全 untouched 的測試集；新 MLP 結果亦不應與歷史表格直接作模型優劣結論。

---

## 限制與後續工作

1. **資料規模**：四部影片、單人標註，統計結論有限。後續需擴大資料集並採雙標註者計算 Cohen's Kappa。
2. **自適應優勢未顯現**：本組歷史結果中自適應融合未勝過靜態融合；視覺可靠度普遍偏低。自適應機制仍是設計假設，需在獨立且多樣的資料上驗證。
3. **LLM 教學階段加權未評估**：本文採 SBERT 模式以維持離線可重現性，LLM 模式的教學階段分類功能尚未正式評估。
4. **尚未進行使用者研究**：實際學習效益待人體實驗驗證。

---

## 目錄結構

```
.
├── README.md                   ← 本文件（研究說明）
├── docs/
│   ├── TANET2026_paper_draft.md    ← 論文草稿
│   └── annotation_guideline.md     ← 在 keytake-ai/docs/
└── keytake-ai/                 ← 系統程式碼（見下方 README）
    ├── README.md               ← 安裝與使用說明
    ├── main.py                 ← 主流程入口
    ├── config.py               ← 所有可調參數
    ├── requirements.txt
    ├── src/                    ← 核心模組
    ├── tools/                  ← 標註工具、評估腳本
    ├── tests/                  ← 單元測試
    ├── data/                   ← 語料庫、評估影片、標註資料
    └── results/                ← 實驗結果報告
```

---

## 技術堆疊

| 類別 | 工具 |
|------|------|
| 語音轉錄 | OpenAI Whisper |
| 語意表徵 | Sentence-BERT（paraphrase-multilingual-MiniLM-L12-v2） |
| 視覺追蹤 | MediaPipe Hand Landmarker |
| 手勢分類 | 自訓練雙層 GRU（PyTorch） |
| 跨模態對齊 | CLIP（openai/clip-vit-base-patch32） |
| OCR | Tesseract（pytesseract） |
| 韻律分析 | librosa |
| 後端 | FastAPI + Celery + Redis |
| 影片處理 | FFmpeg + OpenCV |

---

## 參考文獻

1. Apostolidis et al., "Video summarization using deep neural networks: A survey," *Proc. IEEE*, 2021.
2. Argaw et al., "Towards automated multimodal video summarization: Integrating text, audio and visual cues," *arXiv:2506.23714*, 2025.
3. Zhang et al., "MediaPipe Hands: On-device real-time hand tracking," *arXiv:2006.10214*, 2020.
4. Reimers & Gurevych, "Sentence-BERT," *EMNLP-IJCNLP*, 2019.
5. Radford et al., "Learning transferable visual models from natural language supervision," *ICML*, 2021.
6. Radford et al., "Robust speech recognition via large-scale weak supervision," *arXiv:2212.04356*, 2022.
