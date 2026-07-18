"""
CLIP 視覺-文字對齊評分器

本模組提供 OCR→SBERT 路徑的替代方案，利用 CLIP 模型直接計算影像與文字之間的
語意對齊分數（image-text semantic alignment）。在 OCR 辨識困難的場景下表現更佳：
- 低對比度板書（粉筆字跡模糊）
- 高反光投影片（白板反射）
- 模糊手寫字跡（潦草板書）

原理：CLIP 將影像與文字投射至共同的嵌入空間，透過餘弦相似度衡量跨模態語意關聯性，
無需先經過 OCR 識別再進行文字相似度比對，避免了 OCR 誤辨識造成的錯誤傳播。

參考文獻：
    Radford, A., Kim, J.W., Hallacy, C., et al. (2021).
    "Learning Transferable Visual Models From Natural Language Supervision."
    Proceedings of the 38th International Conference on Machine Learning (ICML).

對應計畫書 4.2 步驟三：視覺特徵提取（跨模態對齊分支）
"""

import logging
from typing import Optional

import numpy as np

# 嘗試載入 CLIP 相關套件（transformers 或 open_clip）
try:
    from transformers import CLIPModel, CLIPProcessor
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# 嘗試載入 PIL（影像轉換所需）
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 從 config 載入設定參數
try:
    from config import CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD
except ImportError:
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CLIP_SCORE_THRESHOLD = 0.25

logger = logging.getLogger(__name__)


class CLIPScorer:
    """
    CLIP 視覺-文字對齊評分器（Singleton 模式）

    使用 OpenAI CLIP 模型計算影像區域（ROI）與文字描述之間的語意相似度分數。
    作為 OCR→SBERT 路徑的補充或替代方案，在 OCR 辨識品質不佳的場景下
    （低對比度、高反光、模糊手寫）提供更穩健的跨模態語意對齊評估。

    使用方式：
        scorer = CLIPScorer.get_instance()
        score = scorer.compute_clip_score(roi_bgr, "微積分 導數 公式")

    Singleton 模式：
        CLIP 模型檔案較大，載入耗時，因此採用單例模式確保全域僅載入一次。
    """

    _instance: Optional["CLIPScorer"] = None

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 CLIP 評分器

        Args:
            model_name: CLIP 模型名稱，預設使用 config.py 中的 CLIP_MODEL_NAME
                        （預設值："openai/clip-vit-base-patch32"，輕量且快速）
        """
        self.model_name = model_name or CLIP_MODEL_NAME
        self.model: Optional["CLIPModel"] = None
        self.processor: Optional["CLIPProcessor"] = None
        self.available = False

        self._load_model()

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "CLIPScorer":
        """
        取得 CLIPScorer 單例實例（Singleton Pattern）

        確保 CLIP 模型全域僅載入一次，節省記憶體與載入時間。

        Args:
            model_name: CLIP 模型名稱（僅首次建立時生效）

        Returns:
            CLIPScorer 全域唯一實例
        """
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load_model(self) -> None:
        """
        載入 CLIP 模型與處理器

        若 transformers 或 torch 未安裝，或模型載入失敗，
        將設定 self.available = False，後續呼叫會返回 0.0。
        """
        if not CLIP_AVAILABLE:
            logger.warning(
                "transformers 或 torch 未安裝，CLIP 評分功能不可用。"
                "請安裝：pip install transformers torch"
            )
            return

        if not PIL_AVAILABLE:
            logger.warning(
                "Pillow 未安裝，CLIP 評分功能不可用。"
                "請安裝：pip install Pillow"
            )
            return

        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            # 設定為評估模式，禁用 dropout
            self.model.eval()
            self.available = True
            logger.info(f"成功載入 CLIP 模型: {self.model_name}")
        except Exception as e:
            logger.warning(
                f"CLIP 模型載入失敗 ({e})，評分功能不可用，將返回 0.0"
            )
            self.available = False

    def _bgr_to_pil(self, roi: np.ndarray) -> "Image.Image":
        """
        將 BGR numpy 陣列轉換為 RGB PIL Image

        OpenCV 預設使用 BGR 色彩空間，CLIP 需要 RGB 格式的 PIL Image。

        Args:
            roi: BGR 格式的 numpy 陣列（來自 OpenCV）

        Returns:
            RGB 格式的 PIL Image
        """
        # BGR → RGB 色彩空間轉換
        rgb_array = roi[:, :, ::-1]
        return Image.fromarray(rgb_array)

    def compute_clip_score(self, roi: np.ndarray, text: str) -> float:
        """
        計算影像區域與文字之間的 CLIP 語意對齊分數

        將 ROI 影像與文字分別編碼至 CLIP 嵌入空間，
        透過餘弦相似度衡量跨模態語意關聯程度。

        Args:
            roi: BGR 格式的影像區域 numpy 陣列（來自 OpenCV 裁切）
            text: 用於比對的文字描述（如 ASR 關鍵字組合）

        Returns:
            正規化後的 CLIP 相似度分數 [0.0, 1.0]
            若模型不可用或輸入無效，返回 0.0
        """
        # 檢查模型是否可用
        if not self.available:
            return 0.0

        # 輸入驗證
        if roi is None or roi.size == 0:
            return 0.0

        if not text or not text.strip():
            return 0.0

        try:
            # 將 BGR 影像轉換為 PIL Image
            pil_image = self._bgr_to_pil(roi)

            # 使用 CLIP 處理器準備模型輸入
            inputs = self.processor(
                text=[text.strip()],
                images=pil_image,
                return_tensors="pt",
                padding=True
            )

            # 禁用梯度計算（推論模式）
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 取得影像與文字的正規化嵌入向量
            image_embeds = outputs.image_embeds  # (1, embed_dim)
            text_embeds = outputs.text_embeds    # (1, embed_dim)

            # 計算餘弦相似度（已由模型內部正規化）
            # logits_per_image 的值域約為 [-100, 100]（乘以 temperature）
            cosine_sim = torch.nn.functional.cosine_similarity(
                image_embeds, text_embeds
            ).item()

            # 將餘弦相似度從 [-1, 1] 正規化至 [0, 1]
            normalized_score = (cosine_sim + 1.0) / 2.0

            # 確保結果在有效範圍內
            return float(np.clip(normalized_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"CLIP 評分計算失敗: {e}")
            return 0.0

    def compute_clip_score_with_keywords(
        self,
        roi: np.ndarray,
        asr_keywords: list[str]
    ) -> float:
        """
        計算影像區域與 ASR 關鍵字列表的 CLIP 語意對齊分數

        將關鍵字列表合併為單一文字後計算 CLIP 分數。
        適用於以語音轉錄關鍵字作為文字端輸入的場景。

        Args:
            roi: BGR 格式的影像區域 numpy 陣列
            asr_keywords: ASR 提取的關鍵字列表
                         （如 ["微積分", "導數", "極限"]）

        Returns:
            正規化後的 CLIP 相似度分數 [0.0, 1.0]
            若關鍵字列表為空或模型不可用，返回 0.0
        """
        if not asr_keywords:
            return 0.0

        # 將關鍵字列表合併為空格分隔的字串
        combined_text = " ".join(kw.strip() for kw in asr_keywords if kw.strip())

        if not combined_text:
            return 0.0

        return self.compute_clip_score(roi, combined_text)

    @property
    def is_available(self) -> bool:
        """檢查 CLIP 模型是否可用"""
        return self.available

    @property
    def score_threshold(self) -> float:
        """取得 CLIP 分數閾值（低於此值視為不相關）"""
        return CLIP_SCORE_THRESHOLD
