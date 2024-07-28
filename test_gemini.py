# import requests
# import json

# # Your Gemini API key
# api_key = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"

# # Example endpoint for the Gemini chatbot API (replace with the actual endpoint)
# base_url = "https://api.gemini.com"  # or the specific chatbot endpoint provided by Gemini
# chatbot_endpoint = "/v1/chatbot/message"  # Replace with the actual chatbot endpoint

# def send_message_to_chatbot(message):
#     url = base_url + chatbot_endpoint

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }

#     payload = {
#         "message": message,
#         # Add any other required parameters
#     }

#     response = requests.post(url, headers=headers, data=json.dumps(payload))

#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)
#         return None

# def chat():
#     print("Start chatting with the Gemini chatbot. Type 'exit' to end the chat.")
#     while True:
#         message = input("You: ")
#         if message.lower() == "exit":
#             break
#         response = send_message_to_chatbot(message)
#         if response:
#             print("Chatbot:", response.get("response", "No response key found in the API response"))
#         else:
#             print("Failed to get a response from the chatbot.")

# if __name__ == "__main__":
#     chat()


# import os
# import time
# import google.generativeai as genai

# # Set your API key here or make sure it's set in the environment variables
# os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
# api_key = os.environ['API_KEY']

# # Configure the API key
# genai.configure(api_key=api_key)

# # Initialize the model
# model = genai.GenerativeModel('gemini-1.5-flash')
# chat = model.start_chat(history=[])

# def send_message_with_retry(chat, message, retries=3, delay=5):
#     for attempt in range(retries):
#         try:
#             response = chat.send_message(message)
#             return response
#         except google.api_core.exceptions.InternalServerError as e:
#             print(f"Error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Exiting.")
#                 return None

# # Chat interaction
# response = send_message_with_retry(chat, 'In one sentence, explain how a computer works to a young child.')
# if response:
#     print(response.text)

# response = send_message_with_retry(chat, 'Okay, how about a more detailed explanation to a high schooler?')
# if response:
#     print(response.text)


import os
import time
import google.generativeai as genai

# Set your API key here or make sure it's set in the environment variables
os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
api_key = os.environ['API_KEY']

# Configure the API key
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])

def send_message_with_retry(chat, message, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = chat.send_message(message)
            return response
        except google.api_core.exceptions.InternalServerError as e:
            print(f"Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Exiting.")
                return None

def interactive_chat():
    print("Start chatting with the Gemini chatbot. Type 'exit' to end the chat.")
    while True:
        message = input("You: ")
        if message.lower() == "exit":
            break
        response = send_message_with_retry(chat, message)
        if response:
            print("Chatbot:", response.text)
        else:
            print("Failed to get a response from the chatbot.")

if __name__ == "__main__":
    interactive_chat()
