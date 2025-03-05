from flask import Flask, request, jsonify, send_from_directory
from intent_predictor import predict_intent, get_response

app = Flask(__name__)

# Route to serve the chatbot HTML file
@app.route('/')
def index():
    return send_from_directory('static', 'chatbot.html')

# Route to handle chatbot messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    
    # Predict intent based on user input
    intent = predict_intent(user_input)
    
    # Get response based on the predicted intent
    response = get_response(intent)
    
    # If response is not clear, use ChatGPT as a fallback
    if response == "Sorry, I didn't understand that.":
        response = get_chatgpt_response(user_input)
    
    return jsonify({'response': response})

# Function to get response from ChatGPT (you can implement this as needed)
def get_chatgpt_response(user_input):
    # Implement interaction with OpenAI's API or any fallback mechanism
    # For example, you can call OpenAI's API to get a response based on user_input
    return "This is a fallback response from ChatGPT."

if __name__ == '__main__':
    app.run(debug=False)
