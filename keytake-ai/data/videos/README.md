# 測試影片資料夾

此資料夾存放用於評估的教學影片。

## 建議資料集（對應計畫書 4.4）

從 YouTube OCW 蒐集 30 部跨學科影片，涵蓋：
- 理工類：線性代數、微積分、資料結構、程式設計
- 人文類：歷史、哲學、文學
- 藝術類：設計、音樂理論

## 視覺場景多樣性（計畫書要求）

| 場景類型 | 說明 | 對應 config |
|---------|------|------------|
| 傳統黑板粉筆 | 白色/彩色粉筆書寫 | DOMAIN_BLACKBOARD |
| 數位觸控白板 | 電子筆跡 | DOMAIN_SLIDES |
| 高反光/低對比 | 光影干擾場景 | DOMAIN_CHALLENGING |

## 推薦 OCW 來源

- [台大開放式課程](https://ocw.aca.ntu.edu.tw/)
- [清大開放式課程](https://ocw.nthu.edu.tw/)
- [MIT OpenCourseWare](https://ocw.mit.edu/)
- [YouTube 學術頻道](https://www.youtube.com/@3blue1brown)

## 命名規範

```
lecture_{序號}_{學科}_{場景類型}.mp4
例：
  lecture_01_linear_algebra_blackboard.mp4
  lecture_02_calculus_slides.mp4
  lecture_03_data_structure_challenging.mp4
```

## 注意事項

- 影片檔案不納入 git 版控（已加入 .gitignore）
- 請確認影片授權允許學術研究使用
- 建議每部影片長度 30~90 分鐘
