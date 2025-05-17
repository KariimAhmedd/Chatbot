import torch
import platform

def get_training_device():
    """Get the appropriate device for training based on system and availability"""
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        # Use MPS (Metal) for Mac
        return torch.device("mps")
    elif torch.cuda.is_available():
        # Use CUDA for other systems with NVIDIA GPU
        return torch.device("cuda")
    else:
        # Fallback to CPU if no GPU available
        return torch.device("cpu")

# Print device information on import
print(f"System: {platform.system()}")
print(f"CUDA (NVIDIA GPU) available: {torch.cuda.is_available()}")
print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
print(f"Training device: {get_training_device()}") 