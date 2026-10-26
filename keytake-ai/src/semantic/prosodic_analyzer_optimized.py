"""從音訊估計片段韻律特徵，輸出 s_prosodic ∈ [0, 1]。"""

from pathlib import Path

import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    from config import (
        PROSODIC_PAUSE_WEIGHT,
        PROSODIC_PITCH_WEIGHT,
        PROSODIC_SPEECH_RATE_WEIGHT,
        PROSODIC_VOLUME_WEIGHT,
    )
except ImportError:
    PROSODIC_SPEECH_RATE_WEIGHT = 0.30
    PROSODIC_VOLUME_WEIGHT = 0.30
    PROSODIC_PITCH_WEIGHT = 0.25
    PROSODIC_PAUSE_WEIGHT = 0.15


class ProsodicAnalyzer:
    """估計語速、音量、音高變化與停頓比例，計算韻律重要性分數。"""

    _FEATURE_KEYS = (
        "speech_rate",
        "volume",
        "pitch_variation",
        "pause_ratio",
    )

    def __init__(
        self,
        sr: int = 22050,
        silence_threshold_db: float = -40.0,
    ) -> None:
        if sr <= 0:
            raise ValueError("sr 必須大於 0")
        self.sr = sr
        self.silence_threshold_db = silence_threshold_db

        weights = np.array(
            [
                PROSODIC_SPEECH_RATE_WEIGHT,
                PROSODIC_VOLUME_WEIGHT,
                PROSODIC_PITCH_WEIGHT,
                PROSODIC_PAUSE_WEIGHT,
            ],
            dtype=float,
        )
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("韻律特徵權重必須是有限的非負數")
        if weights.sum() <= 0:
            raise ValueError("至少要有一個韻律特徵權重大於 0")
        self._weights = weights / weights.sum()

    def analyze_segments(
        self, audio_path: str, segments: list[dict]
    ) -> list[dict]:
        """分析片段並原地加入 s_prosodic 欄位。"""
        if not segments:
            return segments

        if not LIBROSA_AVAILABLE:
            for segment in segments:
                segment["s_prosodic"] = 0.5
            return segments

        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            if audio.size == 0:
                raise ValueError("音訊檔案沒有可用樣本")
        except Exception as exc:
            print(f"[ProsodicAnalyzer] 音訊載入失敗，使用預設分數：{exc}")
            for segment in segments:
                segment["s_prosodic"] = 0.5
            return segments

        raw_features: list[dict[str, float]] = []
        for segment in segments:
            try:
                start = float(segment.get("start", 0))
                end = float(segment.get("end", start))
                if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
                    raw_features.append(self._empty_features())
                    continue

                raw_features.append(
                    self._extract_features(audio, sr, start, end)
                )
            except (TypeError, ValueError, IndexError):
                raw_features.append(self._empty_features())

        normalized = self._normalize_features(raw_features)
        for segment, features in zip(segments, normalized):
            segment["s_prosodic"] = self._compute_importance(features)

        return segments

    @classmethod
    def _empty_features(cls) -> dict[str, float]:
        return {
            "speech_rate": 0.0,
            "volume": 0.0,
            "pitch_variation": 0.0,
            "pause_ratio": 1.0,
        }

    def _extract_features(
        self, y: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> dict[str, float]:
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(y), int(end_sec * sr))
        clip = y[start_sample:end_sample]

        if clip.size < int(sr * 0.1):
            return self._empty_features()

        duration = clip.size / sr
        speech_rate = self._estimate_speech_rate(clip, sr, duration)
        volume = self._compute_volume(clip)
        pitch_variation = self._compute_pitch_variation(clip, sr)
        pause_ratio = self._compute_pause_ratio(clip, sr)

        return {
            "speech_rate": speech_rate,
            "volume": volume,
            "pitch_variation": pitch_variation,
            "pause_ratio": pause_ratio,
        }

    @staticmethod
    def _estimate_speech_rate(
        segment_y: np.ndarray, sr: int, duration: float
    ) -> float:
        """以每秒 onset 數作為發音活動密度的近似值，不等同真正音節語速。"""
        if duration <= 0 or segment_y.size == 0:
            return 0.0
        try:
            onset_env = librosa.onset.onset_strength(y=segment_y, sr=sr)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                backtrack=False,
                units="frames",
            )
            return float(len(onsets) / duration)
        except (ValueError, IndexError, FloatingPointError):
            return 0.0

    @staticmethod
    def _compute_volume(segment_y: np.ndarray) -> float:
        """回傳 RMS 音量；後續會在同一批片段內正規化。"""
        if segment_y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=segment_y)[0]
        return float(np.mean(rms)) if rms.size else 0.0

    @staticmethod
    def _compute_pitch_variation(
        segment_y: np.ndarray, sr: int
    ) -> float:
        """以有聲基頻的相對標準差估計音高變化。"""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                segment_y,
                fmin=librosa.note_to_hz("C2"),
                fmax=min(librosa.note_to_hz("C7"), sr / 2 - 1),
                sr=sr,
            )
            if f0 is None:
                return 0.0

            valid = np.isfinite(f0)
            if voiced_flag is not None:
                valid &= voiced_flag
            voiced_f0 = f0[valid]

            if voiced_f0.size < 2:
                return 0.0

            mean_f0 = float(np.mean(voiced_f0))
            if mean_f0 <= 0:
                return 0.0
            return float(np.std(voiced_f0) / mean_f0)
        except (ValueError, IndexError, FloatingPointError):
            return 0.0

    def _compute_pause_ratio(
        self, segment_y: np.ndarray, sr: int
    ) -> float:
        """依相對於片段峰值的短時能量估計低能量比例。"""
        try:
            rms = librosa.feature.rms(y=segment_y)[0]
            if rms.size == 0 or np.max(rms) <= 1e-10:
                return 1.0

            rms_db = librosa.amplitude_to_db(
                rms,
                ref=float(np.max(rms)),
                top_db=80.0,
            )
            return float(np.mean(rms_db < self.silence_threshold_db))
        except (ValueError, IndexError, FloatingPointError):
            return 0.5

    def _normalize_features(
        self, features_list: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """以同批片段的分位數縮放，降低極端值對結果的影響。"""
        if not features_list:
            return []

        arrays = {
            key: np.array([item[key] for item in features_list], dtype=float)
            for key in self._FEATURE_KEYS
        }
        normalized: dict[str, np.ndarray] = {}

        for key, values in arrays.items():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            low = float(np.percentile(values, 10))
            high = float(np.percentile(values, 90))

            if high - low < 1e-8:
                normalized[key] = np.full(values.shape, 0.5, dtype=float)
            else:
                normalized[key] = np.clip((values - low) / (high - low), 0, 1)

        # 高分代表語速較慢；停頓越少分數越高。
        results = []
        for i in range(len(features_list)):
            results.append(
                {
                    key: float(normalized[key][i])
                    for key in self._FEATURE_KEYS
                }
            )
        return results

    def _compute_importance(self, features: dict[str, float]) -> float:
        """依設定權重加總四項正規化特徵。"""
        components = np.array(
            [
                1.0 - features["speech_rate"],
                features["volume"],
                features["pitch_variation"],
                1.0 - features["pause_ratio"],
            ],
            dtype=float,
        )
        return float(np.clip(np.dot(self._weights, components), 0.0, 1.0))
