# VibeVoice Local (RTX 3060 Edition)

This is a local adaptation of the VibeVoice project, optimized for running on an RTX 3060 (6GB VRAM).

## Features
- **Independent Environment**: Uses `uv` for clean package management.
- **RTX 3060 Optimizations**: 
    - Uses `bfloat16` precision (Ampere architecture).
    - Checks for Flash Attention 2.
    - Trims audio >15s to prevent Out-Of-Memory (OOM) errors.
- **Multi-Speaker & Speed Control**: Includes all extended features from the Colab notebook.

## Prerequisites

1. **NVIDIA Driver**: Ensure you have the latest drivers for your RTX 3060.
2. **FFmpeg**: Required for audio processing.
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: `winget install ffmpeg` (or download from website and add to PATH)
   - **MacOS**: `brew install ffmpeg`
3. **uv**: Fast Python package manager.
   - `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

1. Navigate to this directory:
   ```bash
   cd VibeVoice-Local-RTX3060
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

## Running the App

Run the application using `uv`:

```bash
uv run python src/app.py
```

The Gradio interface will launch at `http://localhost:7860`.

## Notes on VRAM (6GB)

Running a 1.5B TTS model on 6GB VRAM is tight.
- **Audio Trimming**: Input reference audio is automatically trimmed to 15s.
- **Generation Speed**: Might be slower if fallback to system RAM occurs (though we try to avoid it).
- **Concurrency**: Only generates one stream at a time.

## Custom Models

The default model is `vibevoice/VibeVoice-1.5B`. You can change this in `src/config.py` if you want to try the smaller `0.5B` model for better performance:

```python
# src/config.py
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B" 
```
