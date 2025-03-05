from intent_predictor import predict_intent, get_response  # Import functions

def get_chatgpt_response(user_input):
    # Placeholder function for ChatGPT fallback
    return f"ChatGPT response for: {user_input}"

def chatbot():
    """
    A simple chatbot interface that predicts user intent and generates responses.
    If no valid response is found, it falls back to ChatGPT.
    """
    print("Chatbot: Hi buddy, how can I assist you? (Type 'exit' to quit)")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                print("Chatbot: Please say something!")
                continue
            
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break

            # Predict intent and generate response
            intent = predict_intent(user_input)
            response = get_response(intent)

            # Fallback to ChatGPT if no valid response
            if response == "Sorry, I didn't understand that.":
                response = get_chatgpt_response(user_input)
            
            print(f"Chatbot: {response}")
        
        except Exception as e:
            print("Chatbot: Oops! Something went wrong. Please try again.")
            # Log the error (optional)
            print(f"Error: {e}")

# Start the chatbot
if __name__ == "__main__":
    chatbot()


