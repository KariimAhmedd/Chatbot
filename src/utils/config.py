import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Directory paths
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for dir_path in [DATA_DIR, CONFIG_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
CHATBOT_MODEL_PATH = MODELS_DIR / "chatbot_model.pth"
IMAGE_PROCESSOR_PATH = MODELS_DIR / "image_processor.pth"

# Config paths
INTENTS_PATH = CONFIG_DIR / "intents.json"

# Environment variables
def get_api_key():
    """Get the Together AI API key from environment variables."""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        raise ValueError(
            "Please set your Together AI key as an environment variable named 'TOGETHER_API_KEY'"
        )
    return api_key 