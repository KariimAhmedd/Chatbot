import json
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import sys
import traceback
import os
import platform
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from src.models.chatbot import ChatBot
from src.models.image_processor import ImageProcessor
from src.gui.gui import start_gui
from src.utils.config import get_api_key, CHATBOT_MODEL_PATH, IMAGE_PROCESSOR_PATH
from src.utils.device import get_training_device

def train_model(model, train_texts, labels, num_epochs=5, learning_rate=2e-5):
    print("Training model...")
    # Move model to training device (MPS if available)
    training_device = get_training_device()
    model = model.to(training_device)
    print(f"Training on device: {training_device}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for idx, (text, label) in enumerate(zip(train_texts, labels)):
            # Convert inputs to tensor
            inputs = model.tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding='max_length')
            
            # Move inputs and labels to training device
            inputs = {k: v.to(training_device) for k, v in inputs.items()}
            label_tensor = torch.tensor([label], device=training_device)
            
            # Forward pass
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, label_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(train_texts)} examples")
        
        avg_loss = total_loss / len(train_texts)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Move model back to CPU for inference
    model = model.to('cpu')

def setup_environment():
    """Set up environment-specific configurations."""
    if platform.system() == 'Darwin':
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        if 'arm64' in platform.machine():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def main():
    try:
        print("Starting the chatbot initialization...")
        setup_environment()
        
        # Initialize models
        print("Initializing models...")
        model = ChatBot(api_key=get_api_key())
        image_processor = ImageProcessor(api_key=get_api_key())

        # Try to load pre-trained image processor
        if os.path.exists(IMAGE_PROCESSOR_PATH):
            print("Loading pre-trained image processor...")
            try:
                image_processor.load_state_dict(torch.load(IMAGE_PROCESSOR_PATH))
                print("Successfully loaded pre-trained image processor")
            except Exception as e:
                print(f"Could not load image processor: {str(e)}")

        print("Starting GUI...")
        # Start the GUI
        model.eval()
        image_processor.eval()
        start_gui(model, image_processor)

    except Exception as e:
        print("\nAn error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()