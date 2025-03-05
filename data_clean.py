import nltk
import string
import json
import re

nltk.download('punkt')

# Contraction mapping dictionary
CONTRACTIONS = {
    "what's": "what's", "who's": "who's", "it's": "it's", "that's": "that's", "I'm": "I'm", 
    "you're": "you're", "they're": "they're", "we're": "we're", "don't": "don't", "can't": "can't",
    "won't": "won't", "isn't": "isn't", "aren't": "aren't", "didn't": "didn't", "wasn't": "wasn't",
    "weren't": "weren't", "hasn't": "hasn't", "haven't": "haven't", "wouldn't": "wouldn't", "shouldn't": "shouldn't"
}

# Function to expand contractions before tokenizing
def expand_contractions(text):
    # Use regex to replace contractions with the expanded form
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    return text

# Load the intents data from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists to store data
patterns = []
responses = []
intents_list = []

# Iterate through the intents and extract patterns and responses
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        intents_list.append(intent['tag'])

# Manually tokenize and remove punctuation
def preprocess_text(text):
    text = expand_contractions(text)  # Expand contractions before tokenizing
    # Manually tokenize by splitting the text into words
    tokens = text.lower().split()  # This splits by spaces
    # Remove punctuation
    tokens = [word.strip(string.punctuation) for word in tokens]
    return tokens

# Example of preprocessing patterns
preprocessed_patterns = [preprocess_text(pattern) for pattern in patterns]
print(preprocessed_patterns[:5])  # Show first 5 preprocessed patterns
