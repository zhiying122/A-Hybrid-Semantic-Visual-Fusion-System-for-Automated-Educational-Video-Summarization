# KeyTake AI — 原始碼目錄

本目錄包含 KeyTake AI 系統的核心程式碼。完整的系統架構、安裝說明、使用方式、
標註流程、參數設定與版本歷程，請參閱[專案主文件](../README.md)。

## 快速開始

```bash
pip install -r requirements.txt
copy .env.example .env
python main.py "你的教學影片.mp4"
```

## 目錄說明

| 目錄 | 用途 |
|------|------|
| `src/` | 系統核心模組（前處理、語意、視覺、融合、輸出、平台） |
| `tools/` | 標註工具、評估腳本、模型訓練 |
| `tests/` | 單元測試與煙霧測試 |
| `data/` | 語料庫、測試影片、標註資料 |
| `weights/` | 訓練好的模型權重 |
| `models/` | MediaPipe 手部追蹤模型 |
| `results/` | 評估報告輸出 |

## 測試

```bash
python -m pytest tests/ -v
python tests/smoke_test.py
```
