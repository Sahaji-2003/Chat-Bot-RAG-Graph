import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from groq import Groq
import logging

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Ensure you have set these environment variables
wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# Initialize Weaviate client
weaviate_client = weaviate.Client(
    url=wcd_cluster_url,
    auth_client_secret=AuthApiKey(api_key=wcd_api_key),
    additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
)

# Initialize GROQ client
groq_client = Groq(api_key=groq_api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the maximum length for the message to send to Groq API
MAX_MESSAGE_LENGTH = 2048  # Adjust this value based on the Groq API limitations

# Inappropriate keywords for filtering responses
INAPPROPRIATE_KEYWORDS = ["inappropriate_word1", "inappropriate_word2"]  # Add actual keywords

# Endpoint to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    try:
        response = weaviate_client.query.get("Article4", ["content"]) \
                        .with_near_text({"concepts": [user_message]}).with_additional("vector") \
                        .with_limit(1) \
                        .do()
        logging.debug(f"Weaviate response: {response}")
        # console.log({response})
        
        if 'errors' in response:
            logging.error(f"Failed to query Weaviate: {response['errors']}")
            return jsonify({"error": f"Failed to query Weaviate: {response['errors']}"}), 500
        
        closest_text = response["data"]["Get"]["Article4"][0]["content"]
    except Exception as e:
        logging.error(f"Failed to query Weaviate: {e}")
        return jsonify({"error": f"Failed to query Weaviate: {e}"}), 500
    
    try:
        bot_response = get_groq_response(user_message, closest_text)
        bot_response = filter_response(bot_response)  # Filter the response
    except Exception as e:
        logging.error(f"Failed to get response from Groq: {e}")
        return jsonify({"error": f"Failed to get response from Groq: {e}"}), 500

    return jsonify({"response": bot_response})

def get_groq_response(user_message, context_text):
    # Truncate the message if it exceeds the maximum length
    if len(context_text) > MAX_MESSAGE_LENGTH - len(user_message) - 50:
        context_text = context_text[:MAX_MESSAGE_LENGTH - len(user_message) - 50] + '...'
    
    # Add context to the message
    formatted_message = f"User question: {user_message}\nContext: {context_text}\nPlease provide a concise and relevant response."
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": formatted_message,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=150  # Limit response length
    )
    logging.debug(f"Groq response: {chat_completion}")
    return chat_completion.choices[0].message.content

def filter_response(response):
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword in response:
            return "I'm sorry, I cannot provide that information."
    return response

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)




















# import os
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# import weaviate
# from weaviate.auth import AuthApiKey
# from groq import Groq
# import logging

# app = Flask(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # Ensure you have set these environment variables
# wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
# wcd_api_key = os.getenv("WCD_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")
# jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# # Initialize Weaviate client
# weaviate_client = weaviate.Client(
#     url=wcd_cluster_url,
#     auth_client_secret=AuthApiKey(api_key=wcd_api_key),
#     additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
# )

# # Initialize GROQ client
# groq_client = Groq(api_key=groq_api_key)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the maximum length for the message to send to Groq API
# MAX_MESSAGE_LENGTH = 2048  # Adjust this value based on the Groq API limitations

# # Inappropriate keywords for filtering responses
# INAPPROPRIATE_KEYWORDS = ["inappropriate_word1", "inappropriate_word2"]  # Add actual keywords

# # Endpoint to handle chat messages
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')

#     try:
#         response = weaviate_client.query.get("Article4", ["content"]) \
#                         .with_near_text({"concepts": [user_message]}) \
#                         .with_limit(1) \
#                         .do()
#         logging.debug(f"Weaviate response: {response}")
        
#         if 'errors' in response:
#             logging.error(f"Failed to query Weaviate: {response['errors']}")
#             return jsonify({"error": f"Failed to query Weaviate: {response['errors']}"}), 500
        
#         closest_text = response["data"]["Get"]["Article4"][0]["content"]
#     except Exception as e:
#         logging.error(f"Failed to query Weaviate: {e}")
#         return jsonify({"error": f"Failed to query Weaviate: {e}"}), 500
    
#     try:
#         bot_response = get_groq_response(closest_text)
#         bot_response = filter_response(bot_response)  # Filter the response
#     except Exception as e:
#         logging.error(f"Failed to get response from Groq: {e}")
#         return jsonify({"error": f"Failed to get response from Groq: {e}"}), 500

#     return jsonify({"response": bot_response})

# def get_groq_response(message):
#     # Truncate the message if it exceeds the maximum length
#     if len(message) > MAX_MESSAGE_LENGTH:
#         message = message[:MAX_MESSAGE_LENGTH]
    
#     # Add context to the message
#     formatted_message = f"Please provide a concise and relevant response based on the following content: {message}"
    
#     chat_completion = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": formatted_message,
#             }
#         ],
#         model="llama3-8b-8192",
#         max_tokens=150  # Limit response length
#     )
#     logging.debug(f"Groq response: {chat_completion}")
#     return chat_completion.choices[0].message.content

# def filter_response(response):
#     for keyword in INAPPROPRIATE_KEYWORDS:
#         if keyword in response:
#             return "I'm sorry, I cannot provide that information."
#     return response

# if __name__ == '__main__':
#     # Run the Flask app
#     app.run(debug=True)

# import os
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# import weaviate
# from weaviate.auth import AuthApiKey
# from groq import Groq
# import logging

# app = Flask(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # Ensure you have set these environment variables
# wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
# wcd_api_key = os.getenv("WCD_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")
# jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# # Initialize Weaviate client
# weaviate_client = weaviate.Client(
#     url=wcd_cluster_url,
#     auth_client_secret=AuthApiKey(api_key=wcd_api_key),
#     additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
# )

# # Initialize GROQ client
# groq_client = Groq(api_key=groq_api_key)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the maximum length for the message to send to Groq API
# MAX_MESSAGE_LENGTH = 2048  # Adjust this value based on the Groq API limitations

# # Endpoint to handle chat messages
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')

#     try:
#         response = weaviate_client.query.get("Article4", ["content"]) \
#                         .with_near_text({"concepts": [user_message]}) \
#                         .with_limit(1) \
#                         .do()
#         logging.debug(f"Weaviate response: {response}")
        
#         if 'errors' in response:
#             logging.error(f"Failed to query Weaviate: {response['errors']}")
#             return jsonify({"error": f"Failed to query Weaviate: {response['errors']}"}), 500
        
#         closest_text = response["data"]["Get"]["Article4"][0]["content"]
#     except Exception as e:
#         logging.error(f"Failed to query Weaviate: {e}")
#         return jsonify({"error": f"Failed to query Weaviate: {e}"}), 500
    
#     try:
#         bot_response = get_groq_response(closest_text)
#     except Exception as e:
#         logging.error(f"Failed to get response from Groq: {e}")
#         return jsonify({"error": f"Failed to get response from Groq: {e}"}), 500

#     return jsonify({"response": bot_response})

# def get_groq_response(message):
#     # Truncate the message if it exceeds the maximum length
#     if len(message) > MAX_MESSAGE_LENGTH:
#         message = message[:MAX_MESSAGE_LENGTH]
    
#     chat_completion = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": message,
#             }
#         ],
#         model="llama3-8b-8192",
#     )
#     logging.debug(f"Groq response: {chat_completion}")
#     return chat_completion.choices[0].message.content

# if __name__ == '__main__':
#     # Run the Flask app
#     app.run(debug=True)


