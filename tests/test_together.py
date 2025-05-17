import requests
import os
import json

def test_together():
    print("Testing Together AI connection...")
    
    # Get API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("\n✗ Error: TOGETHER_API_KEY environment variable not set")
        print("\nTo fix this:")
        print("1. Go to https://api.together.xyz/signup")
        print("2. Sign up for free (no credit card required)")
        print("3. Get your API key from https://api.together.xyz/settings/api-keys")
        print("4. Set it in your environment with:")
        print("   export TOGETHER_API_KEY='your-api-key'")
        return False
    
    # API endpoint
    api_url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Test payload using Mixtral model (available in free tier)
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": "Say hello and introduce yourself briefly!"}],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        print("\nSending test message...")
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_json = response.json()
            print("\n✓ Connection successful!")
            print("\nModel response:")
            print(response_json['choices'][0]['message']['content'])
            print("\nFree tier info:")
            print("- 5M tokens per month")
            print("- No credit card required")
            print("- Access to multiple models including Mixtral-8x7B and LLaVA")
            return True
        else:
            print(f"\n✗ API request failed with status {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_together() 