import numpy as np
import torch
import librosa
import subprocess
import os

def post_process_audio(input_wav: str, output_mp3: str) -> bool:
    """
    Applies audio post-processing (Denoise -> Compress -> Normalize) and converts to 44.1kHz MP3.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # FFmpeg filter chain:
        # 1. afftdn: FFT-based noise reduction (nf=-25dB floor)
        # 2. acompressor: Soft-knee compression for voice
        # 3. loudnorm: EBU R128 normalization to -16 LUFS
        filter_chain = "afftdn=nf=-25,acompressor=threshold=-12dB:ratio=4:attack=5:release=50:makeup=2,loudnorm=I=-16:TP=-1.5:LRA=11"
        
        command = [
            "ffmpeg", "-y",
            "-i", input_wav,
            "-filter_complex", filter_chain,
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "192k",
            "-f", "mp3",
            output_mp3
        ]
        
        print(f"Running audio post-processing on {input_wav}...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg processing failed: {result.stderr}")
            return False
            
        print(f"Successfully processed audio to {output_mp3}")
        return True
        
    except FileNotFoundError:
        print("FFmpeg not found. Skipping post-processing.")
        return False
    except Exception as e:
        print(f"Error during post-processing: {e}")
        return False

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
