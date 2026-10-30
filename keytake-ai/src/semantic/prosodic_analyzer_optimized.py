"""從音訊估計韻律特徵，並以 prosodic_available 標記分數是否有效。"""

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
    """Extract relative prosodic heuristics; the resulting score is not calibrated importance."""

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
        """Add a prosodic score and an explicit availability mask to each segment."""
        if not segments:
            return segments

        if not LIBROSA_AVAILABLE:
            for segment in segments:
                self._mark_unavailable(segment)
            return segments

        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            if audio.size == 0:
                raise ValueError("音訊檔案沒有可用樣本")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[ProsodicAnalyzer] 音訊載入失敗，使用預設分數：{exc}")
            for segment in segments:
                self._mark_unavailable(segment)
            return segments

        raw_features: list[dict[str, float] | None] = []
        availability = []
        for segment in segments:
            try:
                start = float(segment.get("start", 0))
                end = float(segment.get("end", start))
                if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
                    raw_features.append(None)
                    availability.append(False)
                    continue

                start_sample = max(0, int(start * sr))
                end_sample = min(len(audio), int(end * sr))
                if end_sample - start_sample < int(sr * 0.1):
                    raw_features.append(None)
                    availability.append(False)
                    continue

                raw_features.append(
                    self._extract_features(
                        audio,
                        sr,
                        start,
                        end,
                        text=str(segment.get("text", "") or ""),
                    )
                )
            except (TypeError, ValueError, IndexError):
                raw_features.append(None)
                availability.append(False)
                continue
            availability.append(True)

        # 正規化上下界由本次影片的有效片段估計；結果是片內相對排序，不可跨影片當絕對量尺。
        normalized = self._normalize_features(raw_features)
        for segment, features, is_available in zip(
            segments,
            normalized,
            availability,
        ):
            segment["prosodic_available"] = is_available
            segment["s_prosodic"] = (
                self._compute_importance(features) if is_available else 0.5
            )

        return segments

    @staticmethod
    def _mark_unavailable(segment: dict) -> None:
        segment["s_prosodic"] = 0.5
        segment["prosodic_available"] = False

    @classmethod
    def _empty_features(cls) -> dict[str, float]:
        return {
            "speech_rate": 0.0,
            "volume": 0.0,
            "pitch_variation": 0.0,
            "pause_ratio": 1.0,
        }

    def _extract_features(
        self,
        y: np.ndarray,
        sr: int,
        start_sec: float,
        end_sec: float,
        text: str = "",
    ) -> dict[str, float]:
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(y), int(end_sec * sr))
        clip = y[start_sample:end_sample]

        if clip.size < int(sr * 0.1):
            return self._empty_features()

        duration = clip.size / sr
        speech_rate = self._estimate_speech_rate(text, duration)
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
    def _estimate_speech_rate(text: str, duration: float) -> float:
        """以逐字稿非空白字元數/秒估計語速；不是音節或詞級語速。"""
        if duration <= 0 or not text:
            return 0.0
        character_count = sum(not char.isspace() for char in text)
        return float(character_count / duration)

    @staticmethod
    def _compute_volume(segment_y: np.ndarray) -> float:
        """回傳平均 RMS 音量；後續在同批片段內正規化。"""
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
        if sr <= 0:
            return 0.5
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
        self, features_list: list[dict[str, float] | None]
    ) -> list[dict[str, float]]:
        """以同批片段的 10/90 分位數縮放，降低極端值影響。"""
        if not features_list:
            return []

        valid_indices = [
            index
            for index, features in enumerate(features_list)
            if features is not None
        ]
        arrays = {
            key: np.array(
                [features_list[index][key] for index in valid_indices],
                dtype=float,
            )
            for key in self._FEATURE_KEYS
        }
        normalized: dict[str, np.ndarray] = {}

        for key, values in arrays.items():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            normalized_values = np.full(len(features_list), 0.5, dtype=float)
            low = float(np.percentile(values, 10)) if values.size else 0.0
            high = float(np.percentile(values, 90)) if values.size else 0.0

            if values.size and high - low >= 1e-8:
                normalized_values[valid_indices] = np.clip(
                    (values - low) / (high - low),
                    0,
                    1,
                )
            normalized[key] = normalized_values

        # 以 10/90 分位數縮放後，慢語速、較大音量、較大音高變化、較少停頓分別偏高；
        # 這是表達型態的啟發式組合，不是教學重要性真值。
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
        """依設定權重加總四項正規化特徵，回傳 [0, 1] 的韻律重要性分數。"""
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
