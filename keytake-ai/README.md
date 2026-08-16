# KeyTake AI — 原始碼目錄

本目錄為 KeyTake AI 系統的主要程式碼所在。

完整的安裝說明、使用方式、系統架構與參數設定，請參閱上層的 [專案 README](../README.md)。

---

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 複製環境變數設定
copy .env.example .env

# 處理一部影片
python main.py "你的教學影片.mp4"

# 執行測試
python -m pytest tests/ -v
```

---

## 目錄說明

| 目錄 | 用途 |
|------|------|
| `src/` | 系統核心模組（語意、視覺、融合、輸出、平台） |
| `tools/` | 標註工具、評估腳本、模型訓練 |
| `tests/` | 單元測試與煙霧測試 |
| `data/` | 語料庫、測試影片、Ground Truth 標註 |
| `weights/` | 預訓練模型權重 |
| `models/` | MediaPipe 等外部模型（離線使用） |
