import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the intents data from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists to store data
patterns = []
intents_list = []

# Extract patterns and intents from the data
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents_list.append(intent['tag'])

# Vectorize the patterns using bag-of-words
vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
X = vectorizer.fit_transform(patterns).toarray()

# Encode the intents
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
naive_bayes = MultinomialNB()
svm = SVC(kernel='linear')
logistic_regression = LogisticRegression(max_iter=1000)

# Train and evaluate Multinomial Naive Bayes
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Multinomial Naive Bayes Accuracy: {accuracy_nb:.4f}")

# Train and evaluate Support Vector Machine (SVM)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Support Vector Machine Accuracy: {accuracy_svm:.4f}")

# Train and evaluate Logistic Regression
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
