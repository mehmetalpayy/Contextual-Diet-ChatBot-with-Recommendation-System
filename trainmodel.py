# %%
import nltk
nltk.download('punkt')

# %%
import numpy as np
import random
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torcheval.metrics.functional import multiclass_f1_score
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

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


def tfidf(tokenized_sentences, all_words):
    sentence_words = [stem(word) for word in tokenized_sentences]

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(vocabulary=all_words, lowercase=True)
    tfidf_vector = tfidf_vectorizer.fit_transform([' '.join(sentence_words)])

    # Convert sparse matrix to dense array
    tfidf_array = np.array(tfidf_vector.todense())[0]

    return tfidf_array


def tf(tokenized_sentences, all_words):
    sentence_words = [stem(word) for word in tokenized_sentences]
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

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
# tf için for döngüsü
for (pattern_sentence, tag) in xy:
    # X: term frequency for each pattern_sentence, which is already tokenized
    tf_vector = tf(pattern_sentence, all_words)
    X_train.append(tf_vector)
    # y: PyTorch CrossEntropyLoss needs only class labels(#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)

# bag_of_words için for döngüsü
"""for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence, which is alraedy tokenized
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels(#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)"""
"""#tfidf için for döngüsü
for (pattern_sentence, tag) in xy:
    # X: TF-IDF for each pattern_sentence, which is already tokenized
    tfidf_vector = tfidf(pattern_sentence, all_words)
    X_train.append(tfidf_vector)
    # y: PyTorch CrossEntropyLoss needs only class labels (#'s for all labels), not one-hot
    label = tags.index(tag)
    y_train.append(label)"""

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
# Hyper-parameters- TEST THESE WITH DIFFERENT #'S
num_epochs = 1000
batch_size = 9
learning_rate = 0.001
input_size = len(X_train[0])
# 8 is based on tags so will be different for charlie
hidden_size = 9
output_size = len(tags)


# print(input_size, output_size)

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


train_dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)

test_dataset = ChatDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_list = []
loss_list = []

accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
precision_metric = torchmetrics.Precision(average='weighted', num_classes=output_size, task="multiclass")
recall_metric = torchmetrics.Recall(average='weighted', num_classes=output_size, task="multiclass")
# f1_metric = torchmetrics.F1(average='weighted', num_classes=output_size, task="multiclass")

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

        # Accuracy metriği
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy_metric(predictions, labels)
        total_accuracy += batch_accuracy.item()

        # Precision, Recall, F1 Score metrikleri
        precision_metric.update(predictions, labels)
        recall_metric.update(predictions, labels)
        # f1_metric.update(predictions, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        average_accuracy = total_accuracy / len(train_loader)
        average_precision = precision_metric.compute()
        average_recall = recall_metric.compute()
        # average_f1 = f1_metric.compute()
        f1_tensor = multiclass_f1_score(predictions, labels, num_classes=output_size)

        print("Train Results:")
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {average_accuracy:.4f}, Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, F1-Score: {f1_tensor:.4f}')
        epoch_list.append((epoch + 1))
        loss_list.append(round(loss.item(), 4))

# Test kısmı
model.eval()
test_accuracy = 0
with torch.no_grad():
    all_predictions = []
    all_labels = []
    for (words, labels) in test_loader:
        words = torch.tensor(words, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)

        # Accuracy metriği
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy_metric(predictions, labels)
        test_accuracy += batch_accuracy.item()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

average_test_accuracy = test_accuracy / len(test_loader)
print(f'Test Accuracy: {average_test_accuracy:.4f}')

precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1-Score: {f1:.4f}')

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
ax.set_title('Training Loss for Chatbot Language Detection (TF-IDF)')
#change file name
plt.savefig('training_loss_for_chatbot.png',dpi=300, bbox_inches='tight')

'''''
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []

for (pattern_sentence, tag) in xy:
    tf_vector = tf(pattern_sentence, all_words)
    X.append(tf_vector)
    label = tags.index(tag)
    y.append(label)


le = LabelEncoder()
all_labels = np.array(y)  # Tüm etiketler
le.fit(all_labels)
y_encoded = le.transform(all_labels)


class_counts = {tag: y_encoded.tolist().count(tags.index(tag)) for tag in tags}
print("Class counts:", class_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)


# RandomForestClassifier modeli
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, warm_start=True)

train_accuracy_list = []
train_precision_list = []
train_recall_list = []
train_f1_list = []

test_accuracy_list = []
test_precision_list = []
test_recall_list = []
test_f1_list = []

epochs = 100
print_interval = 10

for epoch in range(1, epochs + 1):
    rf_model.n_estimators += 10
    rf_model.fit(np.array(X_train), np.array(y_train))

    # Eğitim seti üzerinde tahmin
    y_pred_train_rf = rf_model.predict(np.array(X_train))

    train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
    train_precision_rf = precision_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)
    train_recall_rf = recall_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)
    train_f1_rf = f1_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)

    train_accuracy_list.append(train_accuracy_rf)
    train_precision_list.append(train_precision_rf)
    train_recall_list.append(train_recall_rf)
    train_f1_list.append(train_f1_rf)

    # Test seti üzerinde tahmin
    y_pred_test_rf = rf_model.predict(np.array(X_test))

    test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
    test_precision_rf = precision_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)
    test_recall_rf = recall_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)
    test_f1_rf = f1_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)

    test_accuracy_list.append(test_accuracy_rf)
    test_precision_list.append(test_precision_rf)
    test_recall_list.append(test_recall_rf)
    test_f1_list.append(test_f1_rf)

    # Her 10 epoch'ta sonuçları yazdırma
    if epoch % print_interval == 0:
        print(f'Epoch [{epoch}/{epochs}], Train Accuracy: {train_accuracy_rf:.4f}, '
              f'Train Precision: {train_precision_rf:.4f}, '
              f'Train Recall: {train_recall_rf:.4f}, '
              f'Train F1 Score: {train_f1_rf:.4f}')
        print(f'Test Accuracy: {test_accuracy_rf:.4f}, Test Precision: {test_precision_rf:.4f}, '
              f'Test Recall: {test_recall_rf:.4f}, Test F1 Score: {test_f1_rf:.4f}')

# Eğitim sonuçları
print("Final Train Results:")
print(f'Train Accuracy: {train_accuracy_list[-1]:.4f}, Train Precision: {train_precision_list[-1]:.4f}, '
      f'Train Recall: {train_recall_list[-1]:.4f}, Train F1 Score: {train_f1_list[-1]:.4f}')

# Test sonuçları
print("Final Test Results:")
print(f'Test Accuracy: {test_accuracy_list[-1]:.4f}, Test Precision: {test_precision_list[-1]:.4f}, '
      f'Test Recall: {test_recall_list[-1]:.4f}, Test F1 Score: {test_f1_list[-1]:.4f}')


with open('intents.json', 'r') as f:
        intents = json.load(f)
    
    all_words = []
    tags = []
    xy = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))
    
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    X = []
    y = []
    
    for (pattern_sentence, tag) in xy:
        tf_vector = tf(pattern_sentence, all_words)
        X.append(tf_vector)
        label = tags.index(tag)
        y.append(label)
    
    le = LabelEncoder()
    all_labels = np.array(y)  # Tüm etiketler
    le.fit(all_labels)
    y_encoded = le.transform(all_labels)
    
    # Sınıf dengesizliğinin kontrolü
    class_counts = {tag: y_encoded.tolist().count(tags.index(tag)) for tag in tags}
    print("Class counts:", class_counts)
    
    # Oversampling
    oversample = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y_encoded)
    
    class_counts_resampled = {tag: y_resampled.tolist().count(tags.index(tag)) for tag in tags}
    print("Resampled Class counts:", class_counts_resampled)
    
    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, warm_start=True)
    
    cv_results = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
    
    # Sonuçları yazdırma
    print("Cross-validation results with oversampling:", cv_results)
    print("Mean accuracy with oversampling:", np.mean(cv_results))
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    train_accuracy_list = []
    train_precision_list = []
    train_recall_list = []
    train_f1_list = []
    
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    
    epochs = 100
    print_interval = 10
    
    for epoch in range(1, epochs + 1):
        rf_model.n_estimators += 10
        rf_model.fit(np.array(X_train), np.array(y_train))
    
        # Eğitim seti üzerinde tahmin
        y_pred_train_rf = rf_model.predict(np.array(X_train))
    
        train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
        train_precision_rf = precision_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)
        train_recall_rf = recall_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)
        train_f1_rf = f1_score(y_train, y_pred_train_rf, average='weighted', zero_division=1)
    
        train_accuracy_list.append(train_accuracy_rf)
        train_precision_list.append(train_precision_rf)
        train_recall_list.append(train_recall_rf)
        train_f1_list.append(train_f1_rf)
    
        # Test seti üzerinde tahmin
        y_pred_test_rf = rf_model.predict(np.array(X_test))
    
        test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
        test_precision_rf = precision_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)
        test_recall_rf = recall_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)
        test_f1_rf = f1_score(y_test, y_pred_test_rf, average='weighted', zero_division=1)
    
        test_accuracy_list.append(test_accuracy_rf)
        test_precision_list.append(test_precision_rf)
        test_recall_list.append(test_recall_rf)
        test_f1_list.append(test_f1_rf)
    
        # Her 10 epoch'ta sonuçları yazdırma
        if epoch % print_interval == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Accuracy: {train_accuracy_rf:.4f}, '
                  f'Train Precision: {train_precision_rf:.4f}, '
                  f'Train Recall: {train_recall_rf:.4f}, '
                  f'Train F1 Score: {train_f1_rf:.4f}')
            print(f'Test Accuracy: {test_accuracy_rf:.4f}, Test Precision: {test_precision_rf:.4f}, '
                  f'Test Recall: {test_recall_rf:.4f}, Test F1 Score: {test_f1_rf:.4f}')
    
    # Eğitim sonuçları
    print("Final Train Results:")
    print(f'Train Accuracy: {train_accuracy_list[-1]:.4f}, Train Precision: {train_precision_list[-1]:.4f}, '
          f'Train Recall: {train_recall_list[-1]:.4f}, Train F1 Score: {train_f1_list[-1]:.4f}')
    
    # Test sonuçları
    print("Final Test Results:")
    print(f'Test Accuracy: {test_accuracy_list[-1]:.4f}, Test Precision: {test_precision_list[-1]:.4f}, '
          f'Test Recall: {test_recall_list[-1]:.4f}, Test F1 Score: {test_f1_list[-1]:.4f}')
'''
