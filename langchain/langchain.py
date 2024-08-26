import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from groq import Groq
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from flask_cors import CORS

import langchain
langchain.verbose = False
langchain.debug = True
langchain.llm_cache = False



# from langchain.globals import set_debug
# set_debug(True)

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



chat_llm = ChatGroq(temperature=0,api_key=groq_api_key, model="llama3-8b-8192")
# Configure logging
# logging.basicConfig(level=logging.DEBUG)

# Define the maximum length for the message to send to Groq API
MAX_MESSAGE_LENGTH = 2048  # Adjust this value based on the Groq API limitations

# Inappropriate keywords for filtering responses
INAPPROPRIATE_KEYWORDS = ["inappropriate_word1", "inappropriate_word2"]  # Add actual keywords



# Endpoint to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    try:
        response = weaviate_client.query.get("Article5", ["content"]) \
                        .with_near_text({"concepts": [user_message]}) \
                        .with_limit(1) \
                        .do()
        logging.debug(f"Weaviate response: {response}")
        # console.log({response})
        
        if 'errors' in response:
            logging.error(f"Failed to query Weaviate: {response['errors']}")
            return jsonify({"error": f"Failed to query Weaviate: {response['errors']}"}), 500
        
        closest_text = response["data"]["Get"]["Article5"][0]["content"]
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
    prompt1 = ChatPromptTemplate.from_messages([("system", formatted_message), ("human",user_message)])
 
    chain =  prompt1 | chat_llm
    chat_completion = chain.invoke({
                "role": "human",
                "content": formatted_message,
            })
  
    return chat_completion.content

def filter_response(response):
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword in response:
            return " I'm sorry, I cannot provide that information."
    return response

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False)
