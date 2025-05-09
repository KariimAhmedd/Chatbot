# 🤖 KimozChatBot

KimozChatBot is a simple, customizable rule-based chatbot created entirely from scratch using Python. It uses basic NLP techniques and a neural network for intent classification — **no pretrained models** involved!

---

## 🧠 Features

- 🧾 Custom intents via a JSON file
- 💬 Natural conversation with defined patterns/responses
- 🧠 Trained using a basic feedforward neural network
- 📦 No use of pretrained models or external NLP services

---

## 📁 Project Structure

Chatbot_project/
├── main.py         # Core logic: data processing, training, and chatting

├── intents.json    # User-defined intents and responses

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/KariimAhmedd/ChatBot.git
cd ChatBot
pip install numpy nltk torch
python main.py


🗨️ Sample Chat
You: hello
Bot: Hello! How can I assist you?

You: how are you
Bot: Hey, I'm KimozChatBot!

You: goodbye
Bot: Sad to see you go :(

🛠 Customization

To add or change responses:
	1.	Open intents.json
	2.	Add new patterns and responses under an existing or new tag

{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey"],
  "responses": ["Hello!", "Hi there!", "Hey, how can I help?"]
}

👨‍💻 Author

KimozChatBot by Karim Ahmed
Handcrafted with ❤️ — no pretrained magic, just real learning from scratch.
