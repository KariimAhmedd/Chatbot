# AI Chatbot with Text and Image Processing

This project implements a chatbot using Together AI's API, supporting both text chat and image processing capabilities. It uses the Mixtral-8x7B model for text and LLaVA for vision tasks.

## Features
- Text-based chat using Mixtral-8x7B
- Image processing and analysis using LLaVA
- GPU acceleration support (CUDA for NVIDIA, MPS for Apple Silicon)
- Modern GUI interface

## Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Together AI API key:
   - Sign up for a free account at [Together AI](https://www.together.ai/)
   - Get your API key from the dashboard
   - Set it as an environment variable:
     ```bash
     # On Unix/macOS
     export TOGETHER_API_KEY=your_api_key_here
     
     # On Windows (Command Prompt)
     set TOGETHER_API_KEY=your_api_key_here
     
     # On Windows (PowerShell)
     $env:TOGETHER_API_KEY="your_api_key_here"
     ```

5. Run the application:
```bash
python main.py
```

## System Requirements
- Python 3.8 or higher
- For GPU acceleration:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon Mac (M1/M2/M3) for MPS acceleration

## Note
Never commit your API key or any sensitive information to the repository. The `.env` file is ignored by git for security reasons. 