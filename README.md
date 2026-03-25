# KeyTake AI — 自動化教學精華擷取系統

> 多模態融合教學影片自動摘要，整合語音語意分析與手部軌跡視覺偵測，將長篇課堂錄影濃縮為精華片段。

---

## 系統架構

```
輸入影片
  │
  ├─ Path A：視覺事件偵測流
  │    MediaPipe 手部追蹤 → 滯留/指引判斷 → ROI 擷取 → S_visual
  │
  └─ Path B：語意理解流
       Whisper 轉錄 → TF-IDF + Sentence-BERT → S_text
                              │
                    適應性晚期融合
                  Score = α·S_text + β·S_visual
                              │
                    語意感知滑動視窗
                              │
                         摘要影片輸出
```

## 專案結構

```
keytake-ai/
├── main.py                        # 主流程 Pipeline
├── config.py                      # 所有可調參數集中管理
├── requirements.txt
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py        # 步驟一：FFmpeg 轉碼 + 動態降噪
│   ├── semantic/
│   │   ├── transcriber.py         # 步驟二：Whisper 語音轉錄
│   │   └── scorer.py              # 步驟二：TF-IDF + SBERT 雙軌評分
│   ├── visual/
│   │   ├── hand_tracker.py        # 步驟三：MediaPipe 手部軌跡追蹤
│   │   └── visual_scorer.py       # 步驟三：OCR + IoU 降級備援
│   ├── fusion/
│   │   ├── adaptive_fusion.py     # 步驟四：加權融合 + Grid Search + LOOCV
│   │   └── evaluator.py           # 成效評估：Recall / BERTScore / TSR
│   └── platform/
│       └── api.py                 # FastAPI 後端（非同步批次處理）
├── data/
│   └── prompt_corpus/
│       └── corpus.py              # 教學提示語料庫
└── tools/
    ├── annotator.py               # Ground Truth 標註工具（CLI）
    └── export_transcript.py       # 匯出 Whisper 逐字稿供標註使用
```

## 快速開始

### 1. 安裝環境

```bash
pip install -r requirements.txt
```

> 需要額外安裝 FFmpeg：https://ffmpeg.org/download.html
> 需要額外安裝 Tesseract OCR：https://github.com/tesseract-ocr/tesseract

### 2. 執行摘要 Pipeline

```bash
python main.py <影片路徑>
```

### 3. 標註 Ground Truth（雙盲評級）

```bash
# 步驟一：匯出逐字稿
python tools/export_transcript.py lecture.mp4 --output transcript.json

# 步驟二：標註者 A 評分
python tools/annotator.py annotate transcript.json --annotator A

# 步驟三：標註者 B 評分
python tools/annotator.py annotate transcript.json --annotator B

# 步驟四：合併並計算 Cohen's Kappa
python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json
```

### 4. 啟動 Web 平台

```bash
uvicorn src.platform.api:app --reload
```

---

## 評估指標目標

| 指標 | 目標 |
|------|------|
| 重點召回率 (Recall) | > 70% |
| 誤報率 (False Alarm Rate) | < 25% |
| 語意相似度 (BERTScore) | > 0.7 |
| 時間節省率 (Time Saving Rate) | > 50% |

## 核心參數調整

所有參數集中於 `config.py`，主要調校項目：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `WHISPER_MODEL` | Whisper 模型大小 | `base` |
| `ALPHA` / `BETA` | 語意/視覺融合權重 | `0.5 / 0.5` |
| `HAND_VARIANCE_THRESHOLD` | 手部滯留判斷閾值 | `50.0` |
| `SLIDING_WINDOW_MIN_SEC` | 滑動視窗最小保留秒數 | `10` |
| `FUSION_SCORE_THRESHOLD` | 片段保留最低分數 | `0.5` |

## 技術棧

- 語音辨識：[OpenAI Whisper](https://github.com/openai/whisper)
- 語意向量：[Sentence-BERT](https://www.sbert.net/)
- 手部追蹤：[MediaPipe Hands](https://mediapipe.dev/)
- 後端框架：[FastAPI](https://fastapi.tiangolo.com/)
