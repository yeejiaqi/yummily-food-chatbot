
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
from sklearn.preprocessing import label_binarize
import pickle

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents.json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Tokenize and prepare the data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Prepare training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and track accuracy
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total  # Calculate accuracy in 0.00 - 1.00 form

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')

# Final evaluation
model.eval()
all_labels = []
all_predictions = []
all_outputs = []

with torch.no_grad():
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_outputs = np.array(all_outputs)

# Binarize the labels for ROC and AUC calculations
y_test_binarized = label_binarize(all_labels, classes=np.arange(output_size))

# Get ROC curve and AUC for each class (one-vs-rest)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(output_size):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], all_outputs[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], all_outputs[:, i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 7))
for i in range(output_size):
    plt.plot(fpr[i], tpr[i], label=f'Class {tags[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.savefig('roc_curve_nn.png')
plt.show()

# Binary-style confusion matrix for each class
for i, class_name in enumerate(tags):
    print(f"Binary Confusion Matrix for class: '{class_name}'")

    # Convert to binary classification (1 for the current class, 0 for others)
    y_test_binary = (all_labels == i).astype(int)
    y_pred_binary = (all_predictions == i).astype(int)

    # Generate binary confusion matrix
    conf_matrix_binary = confusion_matrix(y_test_binary, y_pred_binary)

    # Extract TP, FP, TN, FN from the binary confusion matrix
    tn, fp, fn, tp = conf_matrix_binary.ravel()

    # Print the binary-style confusion matrix for each class
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'True Negatives (TN): {tn}')

    # Visualize binary confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not ' + class_name, class_name],
                yticklabels=['Not ' + class_name, class_name])
    plt.title(f'Binary Confusion Matrix for "{class_name}"')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'confusion_matrix_class_{i}.png')
    plt.show()

# Final accuracy calculation
final_accuracy = accuracy_score(all_labels, all_predictions)
print(f'Final Accuracy: {final_accuracy:.2f}')

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data_nn.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
