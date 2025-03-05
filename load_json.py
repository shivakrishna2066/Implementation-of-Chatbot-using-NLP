import json

# Load the JSON data
with open('intents.json') as file:
    intents = json.load(file)

# Check the structure of the data
print(intents)
