from sklearn.preprocessing import LabelEncoder
import json

# Load the intents data from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize the list to store intent tags
intents_list = []

# Extract the intent tags from the intents data
for intent in intents:
    intents_list.append(intent['tag'])

# Encode the intents
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents_list)

# Example of encoding a few intents
print("Classes (encoded labels):", label_encoder.classes_)
print("First 5 encoded labels:", y[:5])  # Show first 5 encoded labels
