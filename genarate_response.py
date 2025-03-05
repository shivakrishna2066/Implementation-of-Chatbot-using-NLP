import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Load the intents data from the JSON file
with open('intents.json') as file:
    data = json.load(file)

# Check if the data is a list or dictionary
if isinstance(data, dict) and 'intents' in data:
    intents = data['intents']
else:
    intents = data  # Assuming the data is a list of intents

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

# Function to get response based on the predicted intent
def get_response(intent):
    # Find the responses for the predicted intent
    for i in range(len(intents)):
        if intents[i]['tag'] == intent:
            return random.choice(intents[i]['responses'])

# Example usage
user_input = "Hello, how are you?"
predicted_intent = predict_intent(user_input)
response = get_response(predicted_intent)
print(f"Chatbot Response: {response}")
