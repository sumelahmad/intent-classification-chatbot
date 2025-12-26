import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')


class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatBotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.Y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_inents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for words, tag in self.documents:
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(tag)

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.Y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        x_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.Y, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_moodel(self, model_path, dimension_path):
        with open(dimension_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        return None


def get_stocks():
    stocks = ['apple', 'meta', 'nvda', 'gs', 'msft']
    print(random.sample(stocks, 3))


if __name__ == '__main__':
    assistant = ChatBotAssistant('intents.json', function_mappings={'stocks': get_stocks})
    assistant.parse_inents()
    assistant.load_moodel('Chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message == '/quit':
            break
        print(assistant.process_message(message))
