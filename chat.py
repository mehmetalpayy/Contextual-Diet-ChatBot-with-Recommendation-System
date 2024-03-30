import random #random choice from possible answers
import json
import csv
from fractions import Fraction
import torch
from torch.functional import meshgrid
from recsystem import Recommend

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
convos=[]
charlie=[]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "DietBOT"

list=[]

def get_response(msg):
    lang=msg
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if intent["tag"] == "introduction/help":
                    return (random.choice(intent['responses']))
                    #do all the functions here and utils stuff
                elif intent["tag"] == "diet":
                    return (random.choice(intent['responses']))
                else:
                    return (random.choice(intent['responses']))

    else:
        return "I do not understand..."