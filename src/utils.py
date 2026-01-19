import numpy as np
import torch
import librosa

def adjust_voice_speed(audio_np: np.ndarray, speed_factor: float, sample_rate: int = 24000) -> np.ndarray:
    if speed_factor == 1.0:
        return audio_np

    try:
        # librosa expects float input
        # speed_factor > 1.0 = faster, < 1.0 = slower
        adjusted_audio = librosa.effects.time_stretch(audio_np, rate=speed_factor)
        return adjusted_audio.astype(np.float32)
    except Exception as e:
        print(f"Time stretch failed: {e}")
        return audio_np

def convert_to_16_bit_wav(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.array(data)
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    data = (data * 32767).astype(np.int16)
    return data
