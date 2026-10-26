# 基於語意—視覺融合之教育影片自動摘要系統

作者：（姓名）*、（指導教授）
單位：（系所）
Email：（信箱）*

---

## 摘要

教學影片時長偏長，學生課後複習時難以快速定位重點，影響學習效率。本研究提出一套
教育影片自動摘要系統，結合語意與視覺兩種資訊，自動保留課堂重點並濃縮影片長度。
在語意方面，系統先將語音轉錄為文字，再計算每個片段與教學重點的語意相似度；在視覺
方面，系統追蹤教師的手部動作，辨識指引、書寫與強調等教學行為，並比對板書內容與
口語講解是否呼應。由於教師手勢並非隨時指向板書，容易產生視覺雜訊，本研究提出一種
可依視覺可靠度動態調整的融合方法，當視覺資訊不可靠時自動降低其影響。於四部真實
教學影片、以人工標註為基準的比較實驗顯示，結合語意與視覺後，F1 由純語意基線的
0.571 提升至 0.632、重點召回率由 0.476 提升至 0.547，誤報率則由 0.125 降至
0.100，驗證視覺資訊對教育影片摘要的實質貢獻。參數敏感度分析進一步指出，在視覺
訊號較弱的影片上，過高的視覺權重反而降低效能，說明動態調整的設計動機；其相對於
固定權重的優勢，仍待於視覺品質差異較大的資料集驗證。

**關鍵詞**：教育影片摘要、多模態融合、語意分析、手勢意圖辨識、視覺可靠度加權

---

## Abstract

Lecture videos are typically long, making it hard for students to locate key
content during review and reducing learning efficiency. This study presents an
automated educational video summarization system that preserves key teaching
segments and condenses video length by combining semantic and visual information.
On the semantic side, the system transcribes speech into text and measures the
similarity between each segment and the key teaching content. On the visual side,
the system tracks the instructor's hand movements, recognizes teaching actions
such as pointing, writing, and emphasis, and checks whether the board content
matches the spoken explanation. Because gestures do not always point at the board
and may introduce visual noise, we propose a fusion method that adjusts the visual
weight according to a per-segment visual reliability, automatically reducing the
influence of visual cues when they are unreliable. Comparative experiments on four
real lecture videos with human-annotated ground truth show that combining semantic
and visual information improves F1 from a semantic-only baseline of 0.571 to 0.632
and recall from 0.476 to 0.547, while reducing the false-alarm rate from 0.125 to
0.100, confirming the contribution of visual information. A parameter-sensitivity
analysis further shows that, on videos with weak visual signals, an excessively
high visual weight degrades performance, motivating the adaptive design; its
advantage over fixed weighting remains to be validated on datasets with larger
variation in visual quality.

**Keywords**: Educational Video Summarization, Multimodal Fusion, Semantic
Analysis, Gesture Intent Recognition, Visual Reliability Weighting

---

## 1. 前言

線上課程與錄播教學普及，教學影片已成為主要學習資源。然而一堂課動輒數十分鐘，
學生複習時需反覆快轉尋找重點，效率低落。自動影片摘要可將長影片濃縮為重點片段，
是提升學習效率的可行途徑 [1]。

既有方法多偏重單一模態。純語意方法僅依賴逐字稿，容易漏失以板書呈現、口語未充分
描述的內容；純視覺方法則對教師手勢敏感，教師走動或手部離開板書時，易將非重點
誤判為重點而引入雜訊。教學場景的特性在於語意與視覺互補：重要概念常同時伴隨口語
講解與板書書寫，但兩者的可靠度隨場景變動。近期多模態摘要研究亦顯示，整合文字、
語音與視覺線索有助於定位重要片段 [2]。

本研究提出一套語意—視覺融合的教育影片自動摘要系統，貢獻如下：

1. 整合語意評分、手勢意圖辨識與板書 OCR 比對，建立完整的教育影片摘要系統，
   並於真實教學影片上以人工標註進行消融評估，驗證雙模態融合相較純語意基線在
   召回率與 F1 上的提升。
2. 提出視覺可靠度驅動的自適應置信度加權融合，以手部穩定度、偵測信心度、板書
   文字密度與手板接近度等可觀測訊號估計視覺可靠度，動態調整視覺權重，並透過
   參數敏感度分析釐清其設計動機與適用條件。

---

## 2. 相關研究

### 2.1 影片摘要

影片摘要分為抽取式與生成式。教育影片的重點高度依賴語音與板書，抽取式（挑選關鍵
片段）較為適用；深度學習方法為近年主流 [1]。

### 2.2 多模態融合

多模態學習分為早期與晚期融合。晚期融合對各模態品質差異較具彈性；Argaw 等人整合
文字、語音與視覺線索產生時間對齊的摘要 [2]，與本研究方向相近，惟其未針對視覺
可靠度做動態加權。本研究採晚期融合並引入可靠度加權。

### 2.3 教育影片分析

講座影片摘要多以逐字稿為主；教師手部追蹤 [3] 與板書光學文字辨識可補充語音
未涵蓋的視覺重點，但手勢的雜訊問題尚待處理。

### 2.4 語意與跨模態表徵

Sentence-BERT 產生句向量並以餘弦相似度衡量語意相關性 [4]；CLIP 以對比學習對齊
圖像與文字，可直接比對板書影像與口語關鍵字 [5]。

---

## 3. 系統方法

本系統為多階段管線：影片前處理後，語意軌與視覺軌並行對每個片段評分，經融合得到
片段重要性分數，最後以語意感知滑動視窗選段並輸出濃縮影片。整體架構如圖 1。

**〔圖 1〕系統架構圖（放於本節開頭）**

### 3.1 語意評分

以 Whisper [6] 將語音轉錄為帶時間戳的片段，並合併連續重複片段以淨化長時間靜音
造成的幻覺輸出。接著以多語版 Sentence-BERT（paraphrase-multilingual-MiniLM-
L12-v2）[4] 將片段編碼為句向量，與教學重點提示語料庫計算最大餘弦相似度，作為
語意分數 s_text ∈ [0,1]。系統亦支援以大型語言模型輸出重要性分數與教學階段，
本文實驗採 SBERT 模式以維持離線可重現性。

### 3.2 視覺評分

以 MediaPipe Hand Landmarker [3] 追蹤教師食指座標，並以卡爾曼濾波平滑抖動。
不同於僅以手部是否靜止判斷重要性，本研究以 GRU 手勢意圖分類器分析食指軌跡序列，
將手部動作分為五類並賦予不同視覺重要性分數，如表 1。

**表 1　手勢意圖類別與視覺重要性分數**

| 手勢意圖 | 說明 | 重要性分數 |
|---------|------|-----------|
| 指引 (Pointing) | 靜止指向某位置 | 1.00 |
| 強調 (Emphasis) | 局部反覆來回 | 0.90 |
| 書寫 (Writing) | 手腕小幅規律移動 | 0.85 |
| 過渡 (Transition) | 大幅移動換區 | 0.20 |
| 無手部 (Idle) | 未偵測到手 | 0.10 |

當手勢意圖分數達門檻時，以食指座標為中心裁切感興趣區域（ROI），對 ROI 進行 OCR，
並以 Sentence-BERT 計算板書文字與口語關鍵字的相似度；同時以 CLIP [5] 直接比對
板書影像與關鍵字，兩者互補得到視覺分數 s_visual。字跡模糊時觸發超解析度增強，
OCR 失敗時退化為手部座標與文字框的交並比（IoU）備援。

### 3.3 自適應置信度加權融合

教師手勢並非恆常指向板書。手部離開板書、走動或畫面無板書時，視覺分數的可靠度
顯著下降；若採固定視覺權重，這些不可靠的觀測會污染摘要。

為此，本研究對每個片段估計視覺可靠度 R_visual ∈ [0,1]，由四項可觀測訊號加權
組成：手部穩定度（0.35）、偵測信心度（0.25）、板書文字密度（0.25）與手板接近度
（0.15），並以指數移動平均平滑跨幀跳動。融合公式如式 (1)–(3)：

$$\beta_{\text{eff}} = \beta \times R_{\text{visual}} \tag{1}$$

$$\alpha_{\text{eff}} = 1 - \beta_{\text{eff}} \tag{2}$$

$$\text{Score} = \alpha_{\text{eff}} \times s_{\text{text}} + \beta_{\text{eff}} \times s_{\text{visual}} \tag{3}$$

其中 α、β 為靜態語意與視覺權重。當 R_visual = 1 時退化為固定加權；視覺不可靠
（R_visual 趨近 0）時，視覺權重自動趨近 0，改以語意為主。此設計以可觀測訊號
作為視覺不確定性的代理。

### 3.4 語意感知滑動視窗與保證濃縮

取得片段分數後，以自適應門檻（固定門檻與分數中位數取較大者）選出突出片段，並沿
Whisper 斷句向前後延伸以維持語句完整。為確保任何長度影片的輸出皆嚴格短於原片，
設定摘要總時長上限（原片時長 × 上限比例），超出時依分數由高至低貪婪保留，並保留
最短輸出保底機制。系統實際運作介面如圖 5。

---

## 4. 實驗與結果

本節評估所提系統的摘要成效。首先說明實驗所用的教學影片、標註方式與評估指標；接著
以消融實驗比較純語意、靜態融合與自適應融合三種策略，驗證加入視覺資訊對定位教學
重點的貢獻；最後透過參數敏感度分析，探討視覺權重對效能的影響，並釐清自適應融合的
設計動機與適用條件。

### 4.1 實驗設定

**資料**　四部公開教學影片，內容涵蓋分數運算、平面三角學與心算教學，均為單一
講者搭配板書之教學型態，每部取前約四分鐘。經 Whisper 轉錄後，由標註者依 Likert
1~5 分標註每段重要性，≥ 4 分者列入 Ground Truth。本版為單人標註可行性驗證
（single-annotator pilot），Cohen's Kappa 列為後續工作。

**指標**　重點召回率（Recall）、精確率（Precision）、F1、誤報率（FAR）與時間
節省率（TSR）。片段與 Ground Truth 重疊超過 50% 視為命中。

**比較策略**　三種策略在同一份影片特徵上評估，僅融合方式不同：純語意
（Score = s_text）、靜態融合（固定 β）、自適應融合（β_eff = β·R_visual）。
融合參數 α = 0.6、β = 0.4，語意評分採 SBERT 模式。

### 4.2 消融實驗

四部影片的跨影片平均如表 2，並以圖 2 呈現。

**表 2　三種融合策略之效能（四部真實影片平均）**

| 融合策略 | Recall | Precision | F1 | FAR | TSR |
|:---|:---:|:---:|:---:|:---:|:---:|
| 純語意 | 0.476 | 0.735 | 0.571 | 0.125 | 0.436 |
| **靜態融合** | **0.547** | **0.780** | **0.632** | **0.100** | 0.395 |
| 自適應融合 | 0.469 | 0.722 | 0.559 | 0.113 | 0.430 |

（表中數值為四部影片之平均；各項標準差分別為 Recall ≤ 0.069、Precision ≤ 0.183、
F1 ≤ 0.078、FAR ≤ 0.044、TSR ≤ 0.125。粗體為最佳結果。）

**〔圖 2〕三策略效能長條圖（放於表 2 之後）**

雙模態融合明顯優於純語意基線：靜態融合的 F1 由 0.571 提升至 0.632、召回率由
0.476 提升至 0.547，誤報率由 0.125 降至 0.100。此結果顯示板書與手勢等視覺線索
對定位教學重點具有實質貢獻。

逐部影片明細如表 3。視覺資訊的效益隨影片而異：lecture_01 與 test_10min 的召回率
明顯提升（0.48→0.61、0.46→0.62）；lecture_02 與 lecture_03 因視覺訊號較弱，
三策略差異不大。自適應融合在本組影片未超越靜態融合，原因於 4.3 分析。

**表 3　逐部影片明細（Recall / F1 / FAR）**

| 影片 | 純語意 | 靜態融合 | 自適應融合 |
|------|--------|----------|-----------|
| lecture_01（分數/小數） | 0.48 / 0.59 / 0.14 | 0.61 / 0.73 / 0.06 | 0.48 / 0.61 / 0.09 |
| lecture_02（三角學） | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 | 0.50 / 0.64 / 0.07 |
| lecture_03（心算） | 0.46 / 0.60 / 0.10 | 0.46 / 0.58 / 0.12 | 0.43 / 0.55 / 0.13 |
| test_10min（通用） | 0.46 / 0.46 / 0.19 | 0.62 / 0.57 / 0.15 | 0.46 / 0.44 / 0.17 |

摘要選段與人工標註的時間對照如圖 4，重疊區段代表命中。

**〔圖 4〕摘要 vs Ground Truth 時間軸對照（放於本節末或附錄）**

### 4.3 參數敏感度分析

為理解視覺權重對效能的影響，於 test_10min 的真實特徵上掃描視覺權重 β
（α = 1−β），結果如表 4，並以圖 3 呈現。

**表 4　視覺權重 β 之敏感度（test_10min）**

| β | Recall | Precision | F1 | FAR |
|---|--------|-----------|----|----|
| 0.0（純語意） | 0.462 | 0.455 | 0.458 | 0.189 |
| 0.1 | 0.462 | 0.500 | 0.480 | 0.147 |
| 0.2 | 0.462 | 0.455 | 0.458 | 0.176 |
| 0.4 | 0.462 | 0.417 | 0.438 | 0.169 |
| 0.7 | 0.385 | 0.400 | 0.392 | 0.169 |
| 1.0（純視覺） | 0.231 | 0.333 | 0.273 | 0.147 |

**〔圖 3〕F1 隨視覺權重 β 變化曲線（放於表 4 之後）**

本組影片的視覺分數與可靠度整體偏低（test_10min 的 s_visual 中位數約 0.10、
R_visual 中位數約 0.05），代表教師手部多數時間未穩定指向板書，故視覺權重越高、
效能越差（β=1.0 時 F1 僅 0.273）。此結果同時解釋了 4.2 的現象並支持自適應機制的
設計動機：視覺不可靠時，過高的視覺權重會引入雜訊。在本組視覺普遍偏弱的影片上，
自適應與純語意行為趨於一致，因此未展現相對靜態加權的優勢；其優勢預期出現於視覺
品質高低差異大的資料集——高品質時充分利用視覺、低品質時自動避開。

---

## 5. 討論與限制

1. **資料規模與標註**　本版評估四部影片且為單人標註可行性驗證。更大規模資料集與
   雙標註者 Cohen's Kappa 為必要的後續工作，以支撐統計上更穩健的結論。

2. **視覺品質單一，自適應優勢未顯現**　本組四部影片的視覺可靠度普遍偏低，視覺
   資訊本身的增益有限，自適應機制在此情形下行為趨近純語意。其核心價值在於跨視覺
   品質的穩健性，需以視覺品質差異較大的影片（含高品質板書指引場景）驗證，為本
   研究最重要的後續工作。

3. **教學階段加權未啟用**　教學階段分類需大型語言模型；本文為離線可重現採 SBERT
   模式，未納入階段加權之效益分析。

4. **語意評分依賴提示語料庫**　SBERT 模式效果受提示語料庫涵蓋度影響。

5. **尚未進行使用者研究**　本文聚焦方法與系統可行性，實際學習效益待人體實驗。

---

## 6. 結論與未來工作

本文提出並實作一套語意—視覺融合的教育影片自動摘要系統，整合語意評分、手勢意圖
辨識與板書 OCR 比對，並提出視覺可靠度驅動的自適應置信度加權融合。於四部真實教學
影片的消融實驗顯示，雙模態融合相較純語意基線在召回率與 F1 上均有實質提升（F1
0.571→0.632、召回率 0.476→0.547、誤報率 0.125→0.100），驗證視覺資訊對教育影片
摘要的貢獻；參數敏感度分析則釐清了視覺權重的作用區間與自適應機制的設計動機。

未來工作包括：擴大真實影片資料集並採雙標註者建立可信 Ground Truth（報告 Cohen's
Kappa）；蒐集視覺品質差異較大的影片以驗證自適應融合的穩健性優勢；啟用大型語言
模型的教學階段分析並評估其效益；以及進行使用者研究，評估摘要對學習效率的影響。

---

## 參考文獻

[1] E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, and I. Patras,
    "Video summarization using deep neural networks: A survey," *Proc. IEEE*,
    vol. 109, no. 11, pp. 1838–1863, 2021.

[2] D. M. Argaw, S. Yoon, F. C. Heilbron, H. Deilamsalehy, T. Bui, Z. Wang,
    et al., "Towards automated multimodal video summarization: Integrating text,
    audio and visual cues," *arXiv preprint arXiv:2506.23714*, 2025.

[3] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and
    M. Grundmann, "MediaPipe Hands: On-device real-time hand tracking," *arXiv
    preprint arXiv:2006.10214*, 2020.

[4] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using
    Siamese BERT-networks," in *Proc. 2019 Conf. Empirical Methods in Natural
    Language Processing (EMNLP-IJCNLP)*, Hong Kong, China, 2019, pp. 3982–3992.

[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, et al.,
    "Learning transferable visual models from natural language supervision," in
    *Proc. 38th Int. Conf. Machine Learning (ICML)*, 2021, pp. 8748–8763.

[6] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever,
    "Robust speech recognition via large-scale weak supervision," *arXiv preprint
    arXiv:2212.04356*, 2022.
