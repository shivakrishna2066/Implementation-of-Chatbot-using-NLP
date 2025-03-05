import nltk
from sklearn.feature_extraction.text import CountVectorizer  # Make sure this is imported
import string
import json
import re

# Download necessary NLTK data
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

# Manually tokenize and remove punctuation
def preprocess_text(text):
    text = expand_contractions(text)  # Expand contractions before tokenizing
    # Manually tokenize by splitting the text into words
    tokens = text.lower().split()  # This splits by spaces
    # Remove punctuation
    tokens = [word.strip(string.punctuation) for word in tokens]
    return tokens

# Load the intents data from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists to store data
patterns = []

# Extract patterns from the intents data
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)

# Vectorize the patterns using bag-of-words
vectorizer = CountVectorizer(tokenizer=lambda x: preprocess_text(x))
X = vectorizer.fit_transform(patterns).toarray()

# Get the vocabulary (for older versions of scikit-learn)
vocabulary = vectorizer.get_feature_names()  # Use get_feature_names() for older versions
print(vocabulary[:10])  # Show first 10 words in the vocabulary

