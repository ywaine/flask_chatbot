
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    raise

bot_name = "Sam"

import torch
import random
import logging

def get_response(msg, model, intents, all_words, tags, device, threshold=0.75):
    try:
        # Tokenization
        sentence = tokenize(msg)

        # Bag-of-Words Representation
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Model Prediction
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        # Response Generation
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > threshold:
            tag = tags[predicted.item()]
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])

        # Fallback Response
        return "I do not understand..."
    
    except Exception as e:
        logging.error(f"Error in get_response: {e}")
        return "I encountered an error. Please try again later."

# Example usage:
response = get_response("Hello", model, intents, all_words, tags, device)
print(response)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        try:
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = get_response(sentence)
            print(resp)

        except Exception as e:
            print(f"Error during chat: {e}")
