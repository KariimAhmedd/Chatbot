import random
import json
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')


# ----------- Preprocessing -----------
def tokenize_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(w) for w in words if w.isalpha()]


# ----------- Neural Net -----------
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ----------- Assistant Class -----------
class Assistant:
    def __init__(self, intents_file='intents.json'):
        with open(intents_file) as f:
            self.intents = json.load(f)['intents']
        self.vocabulary = set()
        self.tags = []
        self.data = []

        # Build vocab and training data
        for intent in self.intents:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                words = tokenize_and_lemmatize(pattern)
                self.vocabulary.update(words)
                self.data.append((words, tag))

        self.vocabulary = sorted(list(self.vocabulary))
        self.tags = sorted(list(set(self.tags)))

        X = []
        y = []

        for (words, tag) in self.data:
            X.append(self.bag_of_words(words))
            y.append(self.tags.index(tag))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # Model
        self.model = IntentClassifier(len(self.vocabulary), 64, len(self.tags))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(200):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def process_message(self, message):
        words = tokenize_and_lemmatize(message)
        x = torch.tensor([self.bag_of_words(words)], dtype=torch.float32)
        output = self.model(x)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return "I didn't get that."


# ----------- GUI with Tkinter -----------
assistant = Assistant()

def send():
    user_input = entry.get()
    chat_log.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    if user_input == "/quit":
        root.destroy()
        return

    response = assistant.process_message(user_input)
    chat_log.insert(tk.END, f"Bot: {response}\n\n")

root = tk.Tk()
root.title("Chatbot")

chat_log = tk.Text(root, height=20, width=60)
chat_log.pack()

entry = tk.Entry(root, width=50)
entry.pack(side=tk.LEFT)

send_btn = tk.Button(root, text="Send", command=send)
send_btn.pack(side=tk.LEFT)

root.mainloop()