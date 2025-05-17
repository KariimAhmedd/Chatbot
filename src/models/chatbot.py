import torch
import torch.nn as nn
import requests
import os

class ChatBot(nn.Module):
    def __init__(self, api_key):
        super(ChatBot, self).__init__()
        print("Initializing Together AI model...")
        
        self.api_key = api_key
        
        # API endpoint - using Mixtral model which is available in free tier
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Store conversation history
        self.history = []
        print("Together AI model initialized successfully")
    
    def generate_response(self, message, max_length=150):
        try:
            # Add user message to history
            self.history.append({"role": "user", "content": message})
            
            # Prepare the request payload
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Free tier model
                "messages": self.history,
                "max_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            # Make the API request
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                text_response = response_json['choices'][0]['message']['content']
                # Add bot response to history
                self.history.append({"role": "assistant", "content": text_response})
                return text_response
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                print(f"API Error Details: {error_message}")
                return f"Error: API request failed with status code {response.status_code}"
            
        except Exception as e:
            print(f"Exception details: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def predict(self, text):
        """Compatibility method for the GUI"""
        return self.generate_response(text) 