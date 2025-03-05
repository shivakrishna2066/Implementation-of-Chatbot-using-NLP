import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Function to predict intent
def predict_intent(user_input):
    # Preprocess the user input
    user_input_vector = vectorizer.transform([user_input]).toarray()
    
    # Predict the intent
    predicted_label = model.predict(user_input_vector)[0]
    intent = label_encoder.inverse_transform([predicted_label])[0]
    
    return intent

# Example usage
user_input = "Hello, how are you?"
predicted_intent = predict_intent(user_input)
print(f"Predicted Intent: {predicted_intent}")
