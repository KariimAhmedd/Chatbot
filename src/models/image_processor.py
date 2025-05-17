import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
import base64
from io import BytesIO

class ImageProcessor(nn.Module):
    def __init__(self, api_key):
        super(ImageProcessor, self).__init__()
        print("Initializing Together AI Vision model...")
        
        self.api_key = api_key
        
        # API endpoint for LLaVA model which supports vision
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Keep the image preprocessing for compatibility
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        print("Together AI Vision model initialized successfully")
    
    def process_image(self, image):
        try:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare the request payload using LLaVA model
            payload = {
                "model": "llava-v1.5-13b",  # Free tier vision model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 250,
                "temperature": 0.7
            }
            
            # Make the API request
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json['choices'][0]['message']['content']
            else:
                return f"Error: API request failed with status code {response.status_code}"
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing the image: {str(e)}" 