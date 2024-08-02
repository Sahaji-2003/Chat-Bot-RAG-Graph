
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import time
import sys
import google.generativeai as genai

# Configure the API key for Gemini
os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
api_key = os.environ['API_KEY']
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

class GraphRAG:
    def __init__(self, root_dir, model):
        self.root_dir = root_dir
        self.model = model
        self.python_executable = sys.executable  # Use the Python executable from the virtual environment

    def generate_response(self, query, method='global'):
        index_response = self.run_query_engine(query, method)
        prompt = f"Provide a very concise and summarized response based on the following information: {index_response}"
        response = self.send_message_with_retry(prompt)
        return response

    def run_query_engine(self, query, method):
        command = [
            self.python_executable, "-m", "graphrag.query",
            "--root", self.root_dir,
            "--method", method,
            query
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    def send_message_with_retry(self, message, retries=3, delay=5):
        for attempt in range(retries):
            try:
                chat = self.model.start_chat(history=[])
                response = chat.send_message(message)
                return response.text
            except genai.exceptions.InternalServerError as e:
                print(f"Error: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Exiting.")
                    return "Error: Unable to generate response."

graph_rag = GraphRAG('./ragtest', model)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat_route():
    user_input = request.json['message']
    response = graph_rag.generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

