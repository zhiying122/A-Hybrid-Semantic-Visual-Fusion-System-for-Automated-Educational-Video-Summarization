"""
步驟一：影音前處理與標準化
- 使用 FFmpeg 統一轉換為 MP4
- 動態判斷是否需要頻譜減法（SNR 過低才介入）
對應計畫書 4.2 步驟一
"""

import ffmpeg
import numpy as np
from config import AUDIO_SNR_THRESHOLD, OUTPUT_FORMAT

# Windows 上若 ffmpeg 不在 PATH，指定完整路徑
import os
_FFMPEG_PATH = r"C:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
if os.path.exists(_FFMPEG_PATH):
    os.environ["PATH"] = os.path.dirname(_FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")


def convert_to_mp4(input_path: str, output_path: str) -> str:
    """將任意格式影片轉換為 MP4"""
    ffmpeg.input(input_path).output(output_path).run(overwrite_output=True)
    return output_path


def extract_audio(video_path: str, audio_path: str) -> str:
    """從影片中提取音訊為 WAV（供 Whisper 使用）"""
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True)
    return audio_path


def estimate_snr(audio_samples: np.ndarray) -> float:
    """
    簡易 SNR 估算：以靜音段落（最低 10% 能量）為噪音基準
    回傳 dB 值
    """
    power = audio_samples ** 2
    noise_floor = np.percentile(power, 10)
    signal_power = np.mean(power)
    if noise_floor == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_floor)


def apply_spectral_subtraction(audio_samples: np.ndarray, sr: int) -> np.ndarray:
    """
    頻譜減法降噪（動態選用機制）
    僅在 SNR < AUDIO_SNR_THRESHOLD 時由外部呼叫
    """
    # 簡化實作：使用短時傅立葉轉換估算噪音頻譜並減去
    from numpy.fft import rfft, irfft
    frame_size = int(sr * 0.025)  # 25ms frame
    noise_estimate = np.mean(np.abs(rfft(audio_samples[:frame_size * 10])))
    spectrum = rfft(audio_samples)
    cleaned = np.maximum(np.abs(spectrum) - noise_estimate, 0) * np.exp(1j * np.angle(spectrum))
    return irfft(cleaned, n=len(audio_samples))


def preprocess(input_path: str, output_dir: str) -> dict:
    """
    完整前處理流程，回傳處理後的影片與音訊路徑
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    video_out = os.path.join(output_dir, "video.mp4")
    audio_out = os.path.join(output_dir, "audio.wav")

    convert_to_mp4(input_path, video_out)
    extract_audio(video_out, audio_out)

    # 動態判斷是否需要降噪
    snr = float("inf")
    try:
        with open(audio_out, "rb") as f:
            raw = f.read()[44:]  # 跳過 WAV header
        if len(raw) > 0:
            audio_samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            snr = float(estimate_snr(audio_samples))
            if snr < AUDIO_SNR_THRESHOLD:
                print(f"[Preprocessor] SNR={snr:.1f}dB 過低，啟用頻譜減法")
                cleaned = apply_spectral_subtraction(audio_samples, sr=16000)
                cleaned.astype(np.int16).tofile(audio_out)
            else:
                print(f"[Preprocessor] SNR={snr:.1f}dB 正常，使用原始音訊")
        else:
            print("[Preprocessor] 音訊檔案為空，跳過 SNR 檢查")
    except Exception as e:
        print(f"[Preprocessor] SNR 檢查失敗（{e}），使用原始音訊")

    return {"video": video_out, "audio": audio_out, "snr": snr}
