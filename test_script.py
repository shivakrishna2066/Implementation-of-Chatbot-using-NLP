import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("Python setup is ready!")
print("Pandas version:", pd.__version__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Download the missing resource

text = "Hello! I am learning NLP and building a chatbot."
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(text.lower())
clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

print("Processed Text:", clean_tokens)
