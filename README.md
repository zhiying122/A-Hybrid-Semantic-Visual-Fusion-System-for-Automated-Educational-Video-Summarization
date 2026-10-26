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

整體為多階段管線，語意軌與視覺軌並行處理，最後融合輸出摘要。

```
教學影片
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 步驟一：影音前處理                                    │
│  FFmpeg 格式轉換 → 音訊提取 → 動態 SNR 降噪           │
└─────────────────────┬───────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌────────────────┐       ┌──────────────────────────┐
│  語意軌         │       │  視覺軌                   │
│                │       │                          │
│ Whisper 語音   │       │ MediaPipe 手部追蹤         │
│ 轉錄 + 幻覺    │       │ + 卡爾曼濾波平滑            │
│ 過濾           │       │        ↓                  │
│     ↓          │       │ GRU 手勢意圖分類（5類）     │
│ SBERT 語意     │       │        ↓                  │
│ 相似度評分     │       │ ROI 裁切 → OCR + SBERT     │
│（或 LLM 模式） │       │ 板書語意比對               │
│     ↓          │       │        ↓                  │
│ s_text ∈ [0,1] │       │ CLIP 跨模態對齊            │
│                │       │        ↓                  │
│                │       │ s_visual ∈ [0,1]           │
└────────┬───────┘       └──────────┬───────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────┐
│ 步驟四：自適應置信度加權融合                          │
│                                                     │
│  R_visual = f(手部穩定度, 偵測信心, 板書密度, 接近度) │
│  β_eff = β × R_visual                               │
│  Score = (1 − β_eff) × s_text + β_eff × s_visual   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ 語意感知滑動視窗選段 + 保證濃縮機制                    │
│ → 濃縮影片 + 精華片段索引 + Markdown 學習筆記         │
└─────────────────────────────────────────────────────┘
```

### 核心模組

| 模組 | 位置 | 說明 |
|------|------|------|
| 影音前處理 | `src/preprocessing/preprocessor.py` | FFmpeg 格式轉換、音訊提取、頻譜減法降噪 |
| 語音轉錄 | `src/semantic/transcriber.py` | Whisper 本地/遠端模式，含幻覺過濾 |
| 語意評分 | `src/semantic/scorer.py` | SBERT / LLM / TF-IDF 三模式，LLM 支援教學階段分類 |
| 韻律分析 | `src/semantic/prosodic_analyzer.py` | librosa 語速、音量、音高變化、停頓比例 |
| 手部追蹤 | `src/visual/hand_tracker.py` | MediaPipe Tasks API + 卡爾曼濾波 |
| 手勢分類 | `src/visual/gesture_classifier.py` | 雙層 GRU，輸入食指軌跡序列，輸出 5 類手勢意圖 |
| 視覺評分 | `src/visual/visual_scorer.py` | OCR → SBERT 語意比對，模糊偵測觸發 SRGAN 增強 |
| CLIP 評分 | `src/visual/clip_scorer.py` | 板書影像與口語關鍵字的跨模態語意對齊 |
| 視覺可靠度 | `src/visual/visual_reliability.py` | 四項訊號加權估計 R_visual，EMA 平滑 |
| 自適應融合 | `src/fusion/adaptive_fusion.py` | β_eff = β × R_visual，語意感知滑動視窗 |
| 效能評估 | `src/fusion/evaluator.py` | Recall、Precision、F1、FAR、TSR |
| FastAPI 後端 | `src/platform/api.py` | 影片上傳、非同步處理（Celery + Redis）、摘要下載 |

---

## 手勢意圖分類

GRU 分類器將食指軌跡序列分為五類，各類對應不同視覺重要性分數：

| 手勢意圖 | 說明 | 重要性分數 |
|---------|------|-----------|
| 指引 (Pointing) | 靜止指向某位置 | 1.00 |
| 強調 (Emphasis) | 局部反覆來回 | 0.90 |
| 書寫 (Writing) | 手腕小幅規律移動 | 0.85 |
| 過渡 (Transition) | 大幅移動換區 | 0.20 |
| 無手部 (Idle) | 未偵測到手 | 0.10 |

GRU 架構：雙層，hidden size 64，輸入維度 4（食指 x/y + 移動量），序列長 20 幀，
模型不存在時自動降級為規則型分類器。

---

## 自適應置信度加權融合

視覺可靠度 R_visual ∈ [0,1] 由四項可觀測訊號加權估計：

| 訊號 | 權重 | 說明 |
|------|------|------|
| 手部穩定度 | 0.35 | 食指位移變異數越低越穩定 |
| 偵測信心度 | 0.25 | MediaPipe landmark confidence |
| 板書文字密度 | 0.25 | 文字框面積佔畫面比例（對數壓縮） |
| 手板接近度 | 0.15 | 食指中心點與最近文字框的距離 |

加上指數移動平均（EMA）平滑，避免單幀跳動影響穩定性。

融合公式：

```
β_eff  = β × R_visual
α_eff  = 1 − β_eff
Score  = α_eff × s_text + β_eff × s_visual
```

當 R_visual = 1 時退化為固定加權；視覺不可靠時視覺權重自動趨近 0。

---

## 實驗結果

### 消融實驗（四部真實教學影片平均，α=0.6, β=0.4）

| 融合策略 | Recall | Precision | F1 | FAR | TSR |
|---------|--------|-----------|----|----|-----|
| 純語意 | 0.476 | 0.735 | 0.571 | 0.125 | 0.436 |
| **靜態融合** | **0.547** | **0.780** | **0.632** | **0.100** | 0.395 |
| 自適應融合 | 0.469 | 0.722 | 0.559 | 0.113 | 0.430 |

各項標準差：Recall ≤ 0.069、Precision ≤ 0.183、F1 ≤ 0.078、FAR ≤ 0.044、TSR ≤ 0.125。

靜態融合相較純語意基線：F1 +0.061、Recall +0.071、FAR −0.025。

### 逐部影片明細（Recall / F1 / FAR）

| 影片 | 純語意 | 靜態融合 | 自適應融合 |
|------|--------|----------|-----------|
| lecture_01（分數/小數） | 0.48 / 0.59 / 0.14 | 0.61 / 0.73 / 0.06 | 0.48 / 0.61 / 0.09 |
| lecture_02（三角學） | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 |
| lecture_03（心算） | 0.46 / 0.60 / 0.10 | 0.46 / 0.58 / 0.12 | 0.43 / 0.55 / 0.13 |
| test_10min（通用） | 0.46 / 0.46 / 0.19 | 0.62 / 0.57 / 0.15 | 0.46 / 0.44 / 0.17 |

### 參數敏感度（test_10min，掃描視覺權重 β）

| β | Recall | Precision | F1 | FAR |
|---|--------|-----------|----|----|
| 0.0（純語意） | 0.462 | 0.455 | 0.458 | 0.189 |
| 0.1 | 0.462 | 0.500 | **0.480** | 0.147 |
| 0.4 | 0.462 | 0.417 | 0.438 | 0.169 |
| 1.0（純視覺） | 0.231 | 0.333 | 0.273 | 0.147 |

本組影片 R_visual 中位數約 0.05，視覺訊號整體偏弱，
說明視覺可靠度低時過高的 β 會引入雜訊，支持自適應機制的設計動機。

### 實驗設定說明

- 評估影片：四部公開教學影片，內容涵蓋分數運算、平面三角學與心算，每部取前約四分鐘
- Ground Truth：由標註者依 Likert 1~5 分標註每段重要性，≥ 4 分列入 GT
- 本版為**單人標註可行性驗證（single-annotator pilot）**，Cohen's Kappa 列為後續工作
- 片段與 GT 重疊超過 50% 視為命中

---

## 限制與後續工作

1. **資料規模**：四部影片、單人標註，統計結論有限。後續需擴大資料集並採雙標註者計算 Cohen's Kappa。
2. **自適應優勢未顯現**：本組影片視覺可靠度普遍偏低，自適應融合退化為接近純語意。其優勢預期出現於視覺品質差異大的資料集。
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
