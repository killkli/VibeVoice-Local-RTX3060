import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# We assume the VibeVoice repo was cloned into the project root
VIBEVOICE_REPO_DIR = os.path.join(PROJECT_ROOT, "VibeVoice")
VOICES_DIR = os.path.join(VIBEVOICE_REPO_DIR, "demo", "voices")

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RTX 3060 Optimization Settings
# 6GB VRAM is tight. We need aggressive optimization.
# 1. Use bfloat16 (Ampere supports it and it saves memory/improves stability over float16)
# 2. Use Flash Attention 2 if available
# 3. Trim audio to avoid OOM
MAX_AUDIO_DURATION = 15.0 # seconds

# Model Path (Default)
# DEFAULT_MODEL_PATH = "vibevoice/VibeVoice-1.5B" # 1.5B fits in 6GB roughly. 0.5B is safer.
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "../models/VibeVoice1.5B")) # 1.5B fits in 6GB roughly. 0.5B is safer.
