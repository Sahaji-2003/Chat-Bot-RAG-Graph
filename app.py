



# app.py
# import fitz  # PyMuPDF
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             text += page.get_text()
#     return text

# pdf_files = ["database/ondc1.pdf", "database/ondc2.pdf"]  # Replace with your PDF file paths
# corpus = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# # Load the GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Add padding token to the tokenizer
# tokenizer.pad_token = tokenizer.eos_token

# class GraphRAG:
#     def __init__(self, model, tokenizer, corpus):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.corpus = corpus

#     def generate_response(self, query):
#         input_text = query + " " + self.corpus[0]  # Simplified example

#         # Tokenize and manage length
#         inputs = self.tokenizer.encode_plus(input_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']

#         # Ensure input length + new tokens length does not exceed model's max length
#         max_length = 1024 - 50  # Subtract the length of new tokens
#         if input_ids.size(1) > max_length:
#             input_ids = input_ids[:, :max_length]
#             attention_mask = attention_mask[:, :max_length]

#         # Generate a response with a specific number of new tokens
#         outputs = self.model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=self.tokenizer.eos_token_id)
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

# graph_rag = GraphRAG(model, tokenizer, corpus)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json['message']
#     response = graph_rag.generate_response(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)





# import fitz  # PyMuPDF
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import google.generativeai as genai
# import os
# import time

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             text += page.get_text()
#     return text

# pdf_files = ["database/ondc1.pdf", "database/ondc2.pdf"]  # Replace with your PDF file paths
# corpus = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# # Configure the API key for Gemini
# os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
# api_key = os.environ['API_KEY']
# genai.configure(api_key=api_key)

# # Initialize the model
# model = genai.GenerativeModel('gemini-1.5-flash')
# chat = model.start_chat(history=[])

# class GraphRAG:
#     def __init__(self, chat, corpus):
#         self.chat = chat
#         self.corpus = corpus

#     def generate_response(self, query):
#         # Adjust the prompt to encourage concise responses
#         prompt = f"Provide a concise and summarized response: {query} {self.corpus[0]}"
#         response = self.send_message_with_retry(prompt)
#         return response

#     def send_message_with_retry(self, message, retries=3, delay=5):
#         for attempt in range(retries):
#             try:
#                 response = self.chat.send_message(message)
#                 return response.text
#             except genai.exceptions.InternalServerError as e:
#                 print(f"Error: {e}")
#                 if attempt < retries - 1:
#                     print(f"Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     print("Max retries reached. Exiting.")
#                     return "Error: Unable to generate response."

# graph_rag = GraphRAG(chat, corpus)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/chat', methods=['POST'])
# def chat_route():
#     user_input = request.json['message']
#     response = graph_rag.generate_response(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# import fitz  # PyMuPDF
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import google.generativeai as genai
# import os
# import time

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             text += page.get_text()
#     return text

# pdf_files = ["database/ondc1.pdf", "database/ondc2.pdf"]  # Replace with your PDF file paths
# corpus = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# # Configure the API key for Gemini
# os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
# api_key = os.environ['API_KEY']
# genai.configure(api_key=api_key)

# # Initialize the model
# model = genai.GenerativeModel('gemini-1.5-flash')

# class GraphRAG:
#     def __init__(self, model, corpus):
#         self.model = model
#         self.corpus = corpus

#     def generate_response(self, query):
#         # Start a new chat session with an empty history
#         chat = self.model.start_chat(history=[])
        
#         # Adjust the prompt to encourage concise responses
#         prompt = f"Provide a very concise and summarized response: {query} {self.corpus[0]}"
#         response = self.send_message_with_retry(chat, prompt)
#         return response

#     def send_message_with_retry(self, chat, message, retries=3, delay=5):
#         for attempt in range(retries):
#             try:
#                 response = chat.send_message(message)
#                 return response.text
#             except genai.exceptions.InternalServerError as e:
#                 print(f"Error: {e}")
#                 if attempt < retries - 1:
#                     print(f"Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     print("Max retries reached. Exiting.")
#                     return "Error: Unable to generate response."

# graph_rag = GraphRAG(model, corpus)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/chat', methods=['POST'])
# def chat_route():
#     user_input = request.json['message']
#     response = graph_rag.generate_response(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# import fitz  # PyMuPDF
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import subprocess
# import time
# import google.generativeai as genai

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             text += page.get_text()
#     return text

# # Set up directories and files
# pdf_files = ["database/ondc1.pdf", "database/ondc2.pdf"]  # Replace with your PDF file paths
# input_dir = './ragtest/input'
# os.makedirs(input_dir, exist_ok=True)

# # Extract text and save to input directory
# for pdf_file in pdf_files:
#     text = extract_text_from_pdf(pdf_file)
#     with open(os.path.join(input_dir, os.path.basename(pdf_file).replace('.pdf', '.txt')), 'w', encoding='utf-8') as f:
#         f.write(text)

# # Initialize and run the indexing pipeline
# subprocess.run(["python", "-m", "graphrag.index", "--init", "--root", "./ragtest"])
# subprocess.run(["python", "-m", "graphrag.index", "--root", "./ragtest"])

# # Configure the API key for Gemini
# os.environ['API_KEY'] = "AIzaSyBtO4zrpspHyie3JuXnOuzFMphpeQCPvOk"
# api_key = os.environ['API_KEY']
# genai.configure(api_key=api_key)

# # Initialize the model
# model = genai.GenerativeModel('gemini-1.5-flash')

# class GraphRAG:
#     def __init__(self, root_dir, model):
#         self.root_dir = root_dir
#         self.model = model

#     def generate_response(self, query, method='global'):
#         index_response = self.run_query_engine(query, method)
#         prompt = f"Provide a very concise and summarized response based on the following information: {index_response}"
#         response = self.send_message_with_retry(prompt)
#         return response

#     def run_query_engine(self, query, method):
#         command = [
#             "python", "-m", "graphrag.query",
#             "--root", self.root_dir,
#             "--method", method,
#             query
#         ]
#         result = subprocess.run(command, capture_output=True, text=True)
#         if result.returncode == 0:
#             return result.stdout
#         else:
#             return f"Error: {result.stderr}"

#     def send_message_with_retry(self, message, retries=3, delay=5):
#         for attempt in range(retries):
#             try:
#                 chat = self.model.start_chat(history=[])
#                 response = chat.send_message(message)
#                 return response.text
#             except genai.exceptions.InternalServerError as e:
#                 print(f"Error: {e}")
#                 if attempt < retries - 1:
#                     print(f"Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     print("Max retries reached. Exiting.")
#                     return "Error: Unable to generate response."

# graph_rag = GraphRAG('./ragtest', model)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/chat', methods=['POST'])
# def chat_route():
#     user_input = request.json['message']
#     response = graph_rag.generate_response(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


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

