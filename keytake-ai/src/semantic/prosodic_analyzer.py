"""
步驟 2.5：韻律特徵提取（Prosodic Feature Extraction）
─────────────────────────────────────────────────────────────────────
本模組位於語意分析（步驟二）與視覺特徵提取（步驟三）之間，
透過音訊韻律特徵來輔助判斷教學重點片段。

理論基礎：
  教師在強調重要概念時，通常會表現出以下韻律特徵：
  1. 放慢語速（Speech Rate↓）── 讓學生有時間理解
  2. 提高音量（Volume↑）── 引起注意力
  3. 增大音高變化（Pitch Variation↑）── 語調更具表現力
  4. 減少停頓比例（Pause Ratio↓）── 持續性講解，不中斷

  本模組提取上述四個特徵並計算韻律重要性分數（s_prosodic），
  作為多模態融合的額外輸入維度。

對應計畫書 4.2 步驟二延伸（語意與音訊跨模態補充）
"""

import numpy as np

# ── 嘗試匯入 librosa（音訊分析核心依賴）──────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ── 從 config 匯入可調參數，若無則使用預設值 ─────────────────
try:
    from config import PROSODIC_WEIGHT
except ImportError:
    PROSODIC_WEIGHT = 0.15

try:
    from config import PROSODIC_SPEECH_RATE_WEIGHT
except ImportError:
    PROSODIC_SPEECH_RATE_WEIGHT = 0.3

try:
    from config import PROSODIC_VOLUME_WEIGHT
except ImportError:
    PROSODIC_VOLUME_WEIGHT = 0.3

try:
    from config import PROSODIC_PITCH_WEIGHT
except ImportError:
    PROSODIC_PITCH_WEIGHT = 0.25

try:
    from config import PROSODIC_PAUSE_WEIGHT
except ImportError:
    PROSODIC_PAUSE_WEIGHT = 0.15


class ProsodicAnalyzer:
    """
    韻律特徵分析器

    分析音訊片段的韻律特徵（語速、音量、音高變化、停頓比例），
    並計算韻律重要性分數（s_prosodic），範圍 0~1。

    若 librosa 未安裝，會優雅降級，回傳 0.5 作為所有片段的預設分數。

    Args:
        sr: 音訊取樣率（預設 22050 Hz）
        silence_threshold_db: 靜音判定閾值（分貝），低於此值視為靜音
    """

    def __init__(self, sr: int = 22050, silence_threshold_db: float = -40.0):
        self.sr = sr
        self.silence_threshold_db = silence_threshold_db

        if not LIBROSA_AVAILABLE:
            print("[ProsodicAnalyzer] librosa 未安裝，韻律分析將使用預設分數 0.5")

    def analyze_segments(
        self, audio_path: str, segments: list[dict]
    ) -> list[dict]:
        """
        分析所有片段的韻律特徵，並在每個片段 dict 中加入 s_prosodic 欄位。

        Args:
            audio_path: WAV 音訊檔案路徑
            segments:   片段列表，每個 dict 須包含 'start' 和 'end'（秒）

        Returns:
            加入 s_prosodic 欄位後的片段列表（原地修改並回傳）
        """
        # 若 librosa 不可用，優雅降級
        if not LIBROSA_AVAILABLE:
            for seg in segments:
                seg["s_prosodic"] = 0.5
            return segments

        # 載入音訊
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"[ProsodicAnalyzer] 音訊載入失敗：{e}，使用預設分數")
            for seg in segments:
                seg["s_prosodic"] = 0.5
            return segments

        # 計算每個片段的韻律特徵
        raw_features = []
        for seg in segments:
            start_sec = float(seg.get("start", 0))
            end_sec = float(seg.get("end", start_sec + 1))
            features = self._extract_features(y, sr, start_sec, end_sec)
            raw_features.append(features)

        # 正規化各特徵至 0~1 範圍
        normalized = self._normalize_features(raw_features)

        # 計算韻律重要性分數
        for i, seg in enumerate(segments):
            seg["s_prosodic"] = self._compute_importance(normalized[i])

        return segments

    def _extract_features(
        self, y: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> dict:
        """
        提取單一片段的原始韻律特徵。

        Args:
            y:         完整音訊波形
            sr:        取樣率
            start_sec: 片段起始時間（秒）
            end_sec:   片段結束時間（秒）

        Returns:
            包含 speech_rate, volume, pitch_variation, pause_ratio 的 dict
        """
        # 擷取片段波形
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 避免空片段
        if len(segment_y) < sr * 0.1:  # 少於 0.1 秒
            return {
                "speech_rate": 0.0,
                "volume": 0.0,
                "pitch_variation": 0.0,
                "pause_ratio": 1.0,
            }

        duration = (end_sec - start_sec)

        # ── 語速估計：利用 onset detection 近似音節密度 ──────────
        speech_rate = self._estimate_speech_rate(segment_y, sr, duration)

        # ── 音量（RMS 能量）───────────────────────────────────
        volume = self._compute_volume(segment_y)

        # ── 音高變化（基頻變異係數）──────────────────────────────
        pitch_variation = self._compute_pitch_variation(segment_y, sr)

        # ── 停頓比例（靜音佔比）──────────────────────────────────
        pause_ratio = self._compute_pause_ratio(segment_y, sr)

        return {
            "speech_rate": speech_rate,
            "volume": volume,
            "pitch_variation": pitch_variation,
            "pause_ratio": pause_ratio,
        }

    def _estimate_speech_rate(
        self, segment_y: np.ndarray, sr: int, duration: float
    ) -> float:
        """
        以 onset detection 估計音節密度（每秒音節數）。
        原理：語音中的 onset 事件近似對應音節起始。

        Args:
            segment_y: 片段波形
            sr:        取樣率
            duration:  片段時長（秒）

        Returns:
            每秒音節數估計值
        """
        if duration <= 0:
            return 0.0
        try:
            onset_env = librosa.onset.onset_strength(y=segment_y, sr=sr)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr, backtrack=False
            )
            # 每秒 onset 次數作為語速近似
            return len(onsets) / duration
        except Exception:
            return 0.0

    def _compute_volume(self, segment_y: np.ndarray) -> float:
        """
        計算片段 RMS 能量（均方根振幅）。

        Args:
            segment_y: 片段波形

        Returns:
            RMS 能量值
        """
        rms = librosa.feature.rms(y=segment_y)
        return float(np.mean(rms))

    def _compute_pitch_variation(
        self, segment_y: np.ndarray, sr: int
    ) -> float:
        """
        利用 pyin 演算法提取基頻（F0），計算變異係數。
        變異係數 = 標準差 / 平均值，反映音高變化幅度。

        Args:
            segment_y: 片段波形
            sr:        取樣率

        Returns:
            基頻的變異係數（coefficient of variation）
        """
        try:
            f0, voiced_flag, _ = librosa.pyin(
                segment_y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            # 只取有聲段的 F0 值
            f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(f0_voiced) < 2:
                return 0.0
            mean_f0 = np.mean(f0_voiced)
            if mean_f0 == 0:
                return 0.0
            # 變異係數
            return float(np.std(f0_voiced) / mean_f0)
        except Exception:
            return 0.0

    def _compute_pause_ratio(self, segment_y: np.ndarray, sr: int) -> float:
        """
        計算片段中靜音的比例。
        利用 librosa 的分貝轉換判定每個 frame 是否為靜音。

        Args:
            segment_y: 片段波形
            sr:        取樣率

        Returns:
            靜音比例（0~1），0 表示完全無靜音，1 表示全部靜音
        """
        try:
            # 計算短時能量（分貝）
            rms = librosa.feature.rms(y=segment_y)
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            # 低於閾值的 frame 視為靜音
            silence_frames = np.sum(rms_db < self.silence_threshold_db)
            total_frames = rms_db.shape[-1]
            if total_frames == 0:
                return 1.0
            return float(silence_frames / total_frames)
        except Exception:
            return 0.5

    def _normalize_features(self, features_list: list[dict]) -> list[dict]:
        """
        將所有片段的韻律特徵正規化至 0~1 範圍（min-max 正規化）。

        Args:
            features_list: 各片段原始特徵值列表

        Returns:
            正規化後的特徵列表
        """
        if not features_list:
            return []

        # 收集各特徵的所有值
        keys = ["speech_rate", "volume", "pitch_variation", "pause_ratio"]
        arrays = {k: np.array([f[k] for f in features_list]) for k in keys}

        # Min-Max 正規化
        normalized_arrays = {}
        for k, arr in arrays.items():
            min_val = arr.min()
            max_val = arr.max()
            if max_val - min_val > 1e-8:
                normalized_arrays[k] = (arr - min_val) / (max_val - min_val)
            else:
                # 所有值相同，統一設為 0.5
                normalized_arrays[k] = np.full_like(arr, 0.5)

        # 組裝回 list[dict]
        result = []
        for i in range(len(features_list)):
            result.append({k: float(normalized_arrays[k][i]) for k in keys})
        return result

    def _compute_importance(self, features: dict) -> float:
        """
        根據正規化後的韻律特徵計算重要性分數。

        計分邏輯：
          - 語速越慢 → 分數越高（教師放慢強調）
          - 音量越大 → 分數越高（引起注意）
          - 音高變化越大 → 分數越高（更具表現力）
          - 停頓比例越低 → 分數越高（持續講解，非沈默）

        Args:
            features: 正規化後的特徵 dict

        Returns:
            韻律重要性分數（0~1）
        """
        # 語速：反轉（慢 = 高分）
        speech_rate_score = 1.0 - features["speech_rate"]

        # 音量：正向（大 = 高分）
        volume_score = features["volume"]

        # 音高變化：正向（大變化 = 高分）
        pitch_score = features["pitch_variation"]

        # 停頓比例：反轉（少停頓 = 高分）
        pause_score = 1.0 - features["pause_ratio"]

        # 加權綜合
        importance = (
            PROSODIC_SPEECH_RATE_WEIGHT * speech_rate_score
            + PROSODIC_VOLUME_WEIGHT * volume_score
            + PROSODIC_PITCH_WEIGHT * pitch_score
            + PROSODIC_PAUSE_WEIGHT * pause_score
        )

        # 確保輸出在 0~1 範圍
        return float(np.clip(importance, 0.0, 1.0))
