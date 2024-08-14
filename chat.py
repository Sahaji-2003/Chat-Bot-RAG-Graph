import requests

def chat_with_bot():
    while True:
        user_message = input("You: ")
        if user_message.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        response = requests.post('http://localhost:5000/chat', json={'message': user_message})
        
        if response.status_code == 200:
            bot_response = response.json().get('response')
            print(f"Bot: {bot_response}")
        else:
            error_message = response.json().get('error')
            print(f"Error: {error_message}")

if __name__ == "__main__":
    chat_with_bot()
