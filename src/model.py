import os
import torch
import numpy as np
import librosa
import soundfile as sf
import traceback
import gc
import threading
from typing import Optional, List
from datetime import datetime
from pydub import AudioSegment
import gradio as gr

# Local imports
from .config import MAX_AUDIO_DURATION, VOICES_DIR
from .utils import adjust_voice_speed, convert_to_16_bit_wav

# VibeVoice imports
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.modular.lora_loading import load_lora_assets

class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, adapter_path: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.adapter_path = adapter_path
        self.loaded_adapter_root = None
        self.is_generating = False
        self.stop_generation = False
        self.current_streamer = None
        self.load_model()
        self.setup_voice_presets()
        
    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        load_dtype = torch.float32
        attn_impl = "sdpa"
        
        # GPU capability check for RTX 3060 (Ampere)
        use_flash = False
        if self.device == "cuda":
            try:
                import flash_attn
                # RTX 3060 is Compute Capability 8.6, so it supports Flash Attention 2
                if torch.cuda.get_device_properties(0).major < 8:
                    print("⚠️ GPU capability < 8.0. Forcing SDPA.")
                    use_flash = False
                else:
                    use_flash = True
            except ImportError:
                print("⚠️ flash_attn package not found. Falling back to SDPA.")
                use_flash = False
        
        if use_flash:
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32 
            attn_impl = "sdpa"
            
        print(f"Using attention implementation: {attn_impl} | dtype: {load_dtype}")

        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl,
                device_map=self.device
            )
        except Exception as e:
            print(f"Error loading model with {attn_impl}: {e}")
            if attn_impl == "flash_attention_2":
                print("Retrying with SDPA...")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    attn_implementation="sdpa",
                    device_map=self.device
                )
            else:
                raise e
        
        if self.adapter_path:
            print(f"Loading LoRA from {self.adapter_path}")
            load_lora_assets(self.model, self.adapter_path)
            
        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        print("Model loaded!")

    def setup_voice_presets(self):
        self.voice_presets = {}
        if os.path.exists(VOICES_DIR):
            for f in os.listdir(VOICES_DIR):
                if f.lower().endswith('.wav'):
                    self.voice_presets[os.path.splitext(f)[0]] = os.path.join(VOICES_DIR, f)
        self.available_voices = self.voice_presets
        print(f"Loaded {len(self.available_voices)} voices.")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                
            max_samples = int(MAX_AUDIO_DURATION * target_sr)
            if len(wav) > max_samples:
                print(f"⚠️ Audio too long ({len(wav)/target_sr:.1f}s), trimming to {MAX_AUDIO_DURATION}s to save VRAM")
                wav = wav[:max_samples]
                
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def generate_podcast_streaming(self, num_speakers, script, speakers_list, cfg_scale, 
                                 inference_steps, seed, disable_voice_cloning,
                                 custom_voice_paths: List[str] = [], speed_factor=1.0):
        
        gc.collect()
        torch.cuda.empty_cache()
        
        self.stop_generation = False
        self.is_generating = True
        
        if not script.strip(): raise gr.Error("Please provide a script.")
        script = script.replace("’", "'")
        selected_speakers = speakers_list[:num_speakers]
        
        voice_samples = []
        if not disable_voice_cloning:
            for i, speaker_name in enumerate(selected_speakers):
                custom_path = custom_voice_paths[i] if i < len(custom_voice_paths) else None
                
                if custom_path:
                    print(f"Using Custom Voice File for Speaker {i+1}")
                    audio_data = self.read_audio(custom_path)
                else:
                    audio_data = self.read_audio(self.available_voices[speaker_name])
                
                if len(audio_data) == 0: raise gr.Error(f"Failed to load audio for Speaker {i+1}")
                voice_samples.append(audio_data)
        else:
            voice_samples = None
            
        lines = script.strip().split('\n')
        formatted_lines = []
        for line in lines:
            if not line.strip(): continue
            if line.startswith('Speaker ') and ':' in line:
                formatted_lines.append(line)
            else:
                sid = len(formatted_lines) % num_speakers
                formatted_lines.append(f"Speaker {sid}: {line}")
        formatted_script = '\n'.join(formatted_lines)
        
        processor_kwargs = {
            "text": [formatted_script],
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "voice_samples": [voice_samples] if voice_samples else None
        }
        inputs = self.processor(**processor_kwargs)
        target_device = self.device
        for k, v in inputs.items():
            if torch.is_tensor(v): inputs[k] = v.to(target_device)
            
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None)
        self.current_streamer = audio_streamer
        
        def _gen():
            try:
                generator = torch.Generator(device=target_device).manual_seed(int(seed)) if seed else None
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        generator=generator,
                        audio_streamer=audio_streamer,
                        is_prefill=(not disable_voice_cloning)
                    )
            except Exception as e:
                print(f"Generation error: {e}")
                traceback.print_exc()
                audio_streamer.end()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                
        thread = threading.Thread(target=_gen)
        thread.start()
        
        all_chunks = []
        stream = audio_streamer.get_stream(0)
        
        buffer = []
        buffer_size_samples = 24000 * 1.0 
        current_buffer_len = 0
        
        for chunk in stream:
            if self.stop_generation: 
                audio_streamer.end()
                break
                
            if torch.is_tensor(chunk):
                chunk = chunk.float().cpu().numpy()
            chunk = chunk.astype(np.float32)
            if len(chunk.shape) > 1: chunk = chunk.squeeze()
            
            all_chunks.append(chunk)
            
            buffer.append(chunk)
            current_buffer_len += len(chunk)
            
            if current_buffer_len >= buffer_size_samples:
                buffered_audio = np.concatenate(buffer)
                
                if speed_factor != 1.0:
                    buffered_audio = adjust_voice_speed(buffered_audio, speed_factor)
                
                buffered_audio_16 = convert_to_16_bit_wav(buffered_audio)
                
                yield (24000, buffered_audio_16), None, "Generating..."
                buffer = []
                current_buffer_len = 0
            
        thread.join()
        
        if buffer:
             buffered_audio = np.concatenate(buffer)
             if speed_factor != 1.0:
                 buffered_audio = adjust_voice_speed(buffered_audio, speed_factor)
             yield (24000, convert_to_16_bit_wav(buffered_audio)), None, "Finishing..."
        
        if all_chunks:
            full_audio = np.concatenate(all_chunks)
            if speed_factor != 1.0:
                full_audio = adjust_voice_speed(full_audio, speed_factor)
            
            full_audio_16 = convert_to_16_bit_wav(full_audio)
            
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            # Save to current directory or temp
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            wav_path = os.path.join(output_dir, f"vibevoice_{ts}.wav")
            sf.write(wav_path, full_audio_16, 24000)
            
            mp3_path = wav_path.replace(".wav", ".mp3")
            try:
                AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
                out_path = mp3_path
            except Exception as e:
                print(f"MP3 conversion failed: {e}. Returning WAV.")
                out_path = wav_path
            
            yield None, out_path, "✅ Done! Download ready."
        else:
            yield None, None, "❌ No audio generated."

    def stop_audio_generation(self):
        self.stop_generation = True
        if self.current_streamer: self.current_streamer.end()

    def register_custom_voice(self, name, audio_path):
        if not name or not audio_path:
            return "Please provide both a name and an audio file.", gr.update()
            
        audio = self.read_audio(audio_path)
        if len(audio) == 0:
            return "Error reading audio file.", gr.update()
            
        clean_name = "".join(x for x in name if x.isalnum() or x in "-_")
        save_path = os.path.join(VOICES_DIR, f"{clean_name}.wav")
        sf.write(save_path, audio, 24000)
        
        self.setup_voice_presets()
        
        msg = f"✅ Voice '{clean_name}' saved! It will appear in speaker dropdowns."
        
        new_choices = list(self.available_voices.keys())
        return msg, gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)
