"""影片格式轉換、音訊擷取與條件式降噪。"""

import wave
from pathlib import Path

import ffmpeg
import numpy as np

from config import AUDIO_SNR_THRESHOLD


def convert_to_mp4(input_path: str, output_path: str, fps: int = 30) -> str:
    """轉成固定幀率、相容性較佳的 H.264 MP4。"""
    if fps <= 0:
        raise ValueError("fps 必須大於 0")

    source = Path(input_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到輸入影片：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(source))
        .output(
            str(target),
            format="mp4",
            vcodec="libx264",
            acodec="aac",
            pix_fmt="yuv420p",
            r=fps,
            movflags="+faststart",
        )
        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    )
    return str(target)


def extract_audio(video_path: str, audio_path: str) -> str:
    """擷取 16 kHz、單聲道、16-bit PCM WAV，供語音辨識使用。"""
    source = Path(video_path)
    target = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到輸入影片：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(source))
        .output(
            str(target),
            format="wav",
            ac=1,
            ar=16000,
            acodec="pcm_s16le",
        )
        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    )
    return str(target)


def _frame_audio(
    samples: np.ndarray, frame_size: int, hop_size: int
) -> tuple[np.ndarray, int]:
    """將一維音訊補齊並切成重疊音框。"""
    if samples.ndim != 1:
        raise ValueError("音訊資料必須是一維陣列")
    if samples.size == 0:
        raise ValueError("音訊資料不可為空")

    pad_right = max(frame_size - samples.size, 0)
    padded = np.pad(samples.astype(np.float64, copy=False), (0, pad_right))
    if padded.size < frame_size:
        padded = np.pad(padded, (0, frame_size - padded.size))

    frame_count = 1 + int(np.ceil(max(0, padded.size - frame_size) / hop_size))
    required_size = (frame_count - 1) * hop_size + frame_size
    if padded.size < required_size:
        padded = np.pad(padded, (0, required_size - padded.size))

    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_size)[
        ::hop_size
    ][:frame_count]
    return frames, pad_right


def estimate_snr(audio_samples: np.ndarray, sr: int = 16000) -> float:
    """以音框能量估算 SNR；這是啟用降噪的啟發式指標，不是精密量測。"""
    samples = np.asarray(audio_samples, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("音訊為空或包含非有限數值")
    if sr <= 0:
        raise ValueError("sr 必須大於 0")

    frame_size = max(1, int(sr * 0.025))
    hop_size = max(1, int(sr * 0.010))
    frames, _ = _frame_audio(samples, frame_size, hop_size)
    frame_power = np.mean(frames * frames, axis=1)

    signal_power = float(np.mean(frame_power))
    if signal_power <= 0:
        return 0.0

    noise_power = float(np.percentile(frame_power, 10))
    if noise_power <= 0:
        return float("inf")

    return float(10.0 * np.log10(signal_power / noise_power))


def apply_spectral_subtraction(
    audio_samples: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """使用逐音框頻譜減法降低穩定背景噪音，回傳浮點音訊樣本。"""
    samples = np.asarray(audio_samples, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return samples.copy()
    if not np.all(np.isfinite(samples)):
        raise ValueError("音訊包含非有限數值")
    if sr <= 0:
        raise ValueError("sr 必須大於 0")

    frame_size = max(256, int(sr * 0.025))
    hop_size = max(1, int(sr * 0.010))
    frames, pad_right = _frame_audio(samples, frame_size, hop_size)

    window = np.hanning(frame_size + 1)[:-1]
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))

    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_frame_count = max(1, int(np.ceil(len(frames) * 0.2)))
    noise_indices = np.argsort(frame_rms)[:noise_frame_count]
    noise_profile = np.median(magnitude[noise_indices], axis=0)

    over_subtraction = 1.5
    spectral_floor = 0.02
    cleaned_magnitude = np.maximum(
        magnitude - over_subtraction * noise_profile[None, :],
        spectral_floor * magnitude,
    )
    cleaned_spectrum = cleaned_magnitude * phase
    cleaned_frames = np.fft.irfft(cleaned_spectrum, n=frame_size, axis=1)
    cleaned_frames *= window

    output_size = (len(cleaned_frames) - 1) * hop_size + frame_size
    output = np.zeros(output_size, dtype=np.float64)
    normalization = np.zeros(output_size, dtype=np.float64)

    for index, frame in enumerate(cleaned_frames):
        start = index * hop_size
        output[start : start + frame_size] += frame
        normalization[start : start + frame_size] += window * window

    valid = normalization > 1e-10
    output[valid] /= normalization[valid]
    output[~valid] = 0.0

    end = max(0, output.size - pad_right)
    return output[: min(end, samples.size)]


def preprocess(input_path: str, output_dir: str) -> dict:
    """執行影片轉檔、音訊擷取與條件式降噪。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_out = output_path / "video.mp4"
    audio_out = output_path / "audio.wav"

    convert_to_mp4(input_path, str(video_out))
    extract_audio(str(video_out), str(audio_out))

    snr = float("nan")
    denoised = False

    try:
        with wave.open(str(audio_out), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw_bytes = wav_file.readframes(wav_file.getnframes())

        if channels != 1 or sample_width != 2 or sample_rate != 16000:
            raise ValueError("擷取後音訊格式不是 16 kHz、單聲道、16-bit PCM")

        if raw_bytes:
            audio_samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
            snr = estimate_snr(audio_samples, sr=sample_rate)

            if snr < AUDIO_SNR_THRESHOLD:
                cleaned = apply_spectral_subtraction(audio_samples, sr=sample_rate)
                peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
                if peak > 32767:
                    cleaned = cleaned * (32767.0 / peak)

                cleaned_int16 = np.clip(cleaned, -32768, 32767).astype("<i2")
                with wave.open(str(audio_out), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(cleaned_int16.tobytes())
                denoised = True
    except (wave.Error, OSError, ValueError) as exc:
        print(f"[Preprocessor] SNR 檢查或降噪失敗，保留擷取音訊：{exc}")

    return {
        "video": str(video_out),
        "audio": str(audio_out),
        "snr": snr,
        "denoised": denoised,
    }
