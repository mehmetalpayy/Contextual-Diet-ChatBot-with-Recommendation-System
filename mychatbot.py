# %%
import nltk
nltk.download('punkt')

# %%
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet
import pdb

def bow_with_tfidf(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(vocabulary=all_words, lowercase=True)
    tfidf_vector = tfidf_vectorizer.fit_transform([' '.join(sentence_words)])
    
    # Convert sparse matrix to dense array
    tfidf_array = np.array(tfidf_vector.todense())[0]
    
    return tfidf_array

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 200

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.weight'] = "semibold"

# %%
from collections import Counter

def tf(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    word_counts = Counter(sentence_words)
    
    # TF hesaplama
    tf_array = [word_counts[word] if word in word_counts else 0 for word in all_words]
    
    return np.array(tf_array)

# %%
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# create training data
X_train = []
y_train = []
#tf için for döngüsü
"""for (pattern_sentence, tag) in xy:
    # X: term frequency for each pattern_sentence, which is already tokenized
    tf_vector = tf(pattern_sentence, all_words)
    X_train.append(tf_vector)
    # y: PyTorch CrossEntropyLoss needs only class labels(#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)"""
    
#bag_of_words için for döngüsü
"""for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence, which is alraedy tokenized
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels(#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)"""
#tfidf için for döngüsü
for (pattern_sentence, tag) in xy:
    # X: TF-IDF for each pattern_sentence, which is already tokenized
    tfidf_vector = bow_with_tfidf(pattern_sentence, all_words)
    X_train.append(tfidf_vector)
    # y: PyTorch CrossEntropyLoss needs only class labels (#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Hyperparameters- TEST THESE WITH DIFFERENT #'S
num_epochs = 1000
batch_size = 9
learning_rate = 0.001
input_size = len(X_train[0])
# 8 is based on tags so will be different for charlie
hidden_size = 9
output_size = len(tags)
#print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    # support indexing such that dataset[i] can be used to get i'th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
"""class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i'th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples"""

# Eğitim veri setini DataLoader ile yaratın
train_dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)

# Test veri setini DataLoader ile yaratın
test_dataset = ChatDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

"""dataset = ChatDataset()
#used pytorch because can automatically iterate over code and use as batch training
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_list=[]
loss_list=[]

"""accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
# Train the model
for epoch in range(num_epochs):
    total_accuracy = 0
    for (words, labels) in train_loader:
        words = torch.tensor(words, dtype=torch.float32).to(device)  # Dönüşüm eklenmiş burası
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Accuracy metriğini güncelle
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy_metric(predictions, labels)

        total_accuracy += batch_accuracy.item()  # Batch doğruluk değerini toplam doğruluk değerine ekle
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #pdb.set_trace()
    if (epoch+1) % 100 == 0:
        average_accuracy = total_accuracy / len(train_loader)  # Toplam doğruluk değerini ortalama
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {average_accuracy:.4f}')
        epoch_list.append((epoch+1))   
        loss_list.append(round(loss.item(),4))


print(f'final loss: {loss.item():.4f}')"""
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
precision_metric = torchmetrics.Precision(average='weighted', num_classes=output_size, task="multiclass")
recall_metric = torchmetrics.Recall(average='weighted', num_classes=output_size, task="multiclass")

# Train the model
for epoch in range(num_epochs):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for (words, labels) in train_loader:
        words = torch.tensor(words, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Accuracy metriğini güncelle
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy_metric(predictions, labels)
        total_accuracy += batch_accuracy.item()

        # Precision, Recall, F1 Score metriklerini güncelle
        precision_metric.update(predictions, labels)
        recall_metric.update(predictions, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        average_accuracy = total_accuracy / len(train_loader)
        average_precision = precision_metric.compute()
        average_recall = recall_metric.compute()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {average_accuracy:.4f}, Precision: {average_precision:.4f}, Recall: {average_recall:.4f}')
        epoch_list.append((epoch+1))   
        loss_list.append(round(loss.item(), 4))

print(f'Final loss: {loss.item():.4f}')

# %%
# Modelinizi test verisi üzerinde değerlendirin
model.eval()
test_accuracy = 0
with torch.no_grad():
    all_predictions = []
    all_labels = []
    for (words, labels) in test_loader:
        words = torch.tensor(words, dtype=torch.float32).to(device)  # Dönüşüm eklenmiş burası
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)

        # Accuracy metriğini güncelle
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy_metric(predictions, labels)
        test_accuracy += batch_accuracy.item()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

average_test_accuracy = test_accuracy / len(test_loader)
print(f'Test Accuracy: {average_test_accuracy:.4f}')

# Calculate additional metrics
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# %%
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.plot(epoch_list, loss_list, linewidth=6, color='red')
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Loss")
ax.set_title('Training Loss for Chatbot Language Detection')
#change file name
plt.savefig('training_loss_for_chatbot.png',dpi=300, bbox_inches='tight')

# %%
