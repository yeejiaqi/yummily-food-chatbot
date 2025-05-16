import numpy as np
import random
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk_utils import bag_of_words, tokenize, stem


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

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=100)

model = SVC(kernel='linear')

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 1:}")


import joblib
joblib.dump(model, 'svm_model.pkl')


data = {
    "all_words": all_words,
    "tags": tags
}

with open("model_data.json", "w") as f:
    json.dump(data, f)

print("Model training complete and saved.")
