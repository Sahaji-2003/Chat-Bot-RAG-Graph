# import os
# import json
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from flask_cors import CORS
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# import logging

# # Load environment variables
# load_dotenv()
# jina_api_key = "jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El"
# ES_URL = os.getenv("ES_URL")
# ES_API_KEY = os.getenv("ES_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

# model_name = "BAAI/bge-small-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Initialize Groq chat model
# chat_llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

# # Store chat history
# chat_history_store = {}

# # Define the maximum length for the message to send to Groq API
# MAX_MESSAGE_LENGTH = 2048

# # Initialize embeddings and vector store
# # embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
# vectorstore = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="langchain_index_recursive_bge",
#     embedding=embeddings,
#     es_api_key=ES_API_KEY
# )


# # Initialize Groq chat model
# chat_llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

# document_content_description = (
#     "This collection contains various documents related to real estate transactions, "
#     "including lease agreements, tenant applications, and property management reports. "
#     "Key attributes include tenant names, building names, lease dates, and document sources."
# )


# # Define the maximum length for the message to send to Groq API
# MAX_MESSAGE_LENGTH = 2048

# # Function to perform step-back query expansion
# def step_back_query_expansion(query):
#     # Step 1: Generate a higher-level understanding of the query
#     high_level_query = chat_llm.invoke(f"This is the description of the content: {document_content_description}\n"
#                                       f"- What is the broader concept or main idea behind the query: '{query}'?\n"
#                                       f"Consider the context of real estate and document types. Note: Just give the High-level-query")

#     # Step 2: Expand the original query with related terms, synonyms, and specific context
#     expanded_queries = chat_llm.invoke(f"Expand the following high-level query into multiple queries with related terms, "
#                                       f"synonyms, and context-specific words, considering real estate transactions and "
#                                       f"document attributes. Each query should capture different aspects or angles: {high_level_query}\n"
#                                       f"Note: Provide all the expanded queries in a single text block")


#     return expanded_queries.content

# # Function to perform a search with the expanded query
# def perform_search(query, limit=5):
#     # Perform step-back query expansion
#     expanded_query = step_back_query_expansion(query)
#     print(f"Expanded Query: {expanded_query}")

#     # Perform the search using the expanded query
#     results = vectorstore.similarity_search(expanded_query, k=limit, search_params={"similarity": "cosine"})

#     return results
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')
#     session_id = request.json.get('session_id')  # Get session ID from request

#     # Initialize chat history for the session if it doesn't exist
#     if session_id not in chat_history_store:
#         chat_history_store[session_id] = []

#     # Append the user message to the chat history
#     chat_history_store[session_id].append({"role": "user", "content": user_message})

#     try:
#         # Perform search with step-back query expansion (implement this function as needed)
#         results = perform_search(user_message)

#         # Extract relevant context from results
#         context_text = " ".join([result.page_content for result in results])

#         # Get response from Groq model
#         bot_response = get_groq_response(user_message, context_text, session_id)
        
#         # Append the bot response to the chat history
#         chat_history_store[session_id].append({"role": "bot", "content": bot_response})

#     except Exception as e:
#         return jsonify({"error": f"Error during processing: {e}"}), 500

#     return jsonify({"response": bot_response})

# def get_groq_response(user_message, context_text, session_id):
#     try:
#         # Truncate the message if it exceeds the maximum length
#         if len(context_text) > MAX_MESSAGE_LENGTH - len(user_message) - 50:
#             context_text = context_text[:MAX_MESSAGE_LENGTH - len(user_message) - 50] + '...'

#         # Prepare the chat history for context
#         chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history_store[session_id]])

#         # Add context to the message
#         formatted_message = f"User question: {user_message}\nContext: {context_text}\nChat History:\n{chat_history}\nPlease provide a concise and relevant response."
#         prompt1 = ChatPromptTemplate.from_messages([("system", formatted_message), ("human", user_message)])

#         # Chain the prompt to the chat LLM
#         chain = prompt1 | chat_llm
#         chat_completion = chain.invoke({
#             "role": "human",
#             "content": formatted_message,
#         })

#         return chat_completion.content
    
#     except Exception as e:
#         return f"An error occurred while processing your request: {str(e)}"

# if __name__ == '__main__':
#     app.run(debug=False)








import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the embedding model
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Groq chat model
chat_llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

# Store chat history
chat_history_store = {}

# Define the maximum length for the message to send to Groq API
MAX_MESSAGE_LENGTH = 2048

# Initialize vector store
vectorstore = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive_bge",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)

retriever = vectorstore.as_retriever()

# Document content description
document_content_description = (
    "This collection contains various documents related to real estate transactions, "
    "including lease agreements, tenant applications, and property management reports. "
    "Key attributes include tenant names, building names, lease dates, and document sources."
)

# Contextualization prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat_llm, 
    retriever, 
    contextualize_q_prompt
)

# QA System Prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(chat_llm, qa_prompt)

# Create the Retrieval-Augmented Generation (RAG) chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to perform a search
def perform_search(query, limit=5):
    results = vectorstore.similarity_search(query, k=limit, search_params={"similarity": "cosine"})
    return results

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = request.json.get('session_id', 'new_id')  # Get session ID from request
    
    # Initialize chat history for the session if it doesn't exist
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    
    # Append the user message to the chat history
    chat_history_store[session_id].append({"role": "user", "content": user_message})
    
    try:
        # Perform search
        results = perform_search(user_message)
        
        # Extract relevant context from results
        context_text = " ".join([result.page_content for result in results])
        
        # Get response from Groq model
        bot_response = get_groq_response(user_message, context_text, session_id)
        
        # Append the bot response to the chat history
        chat_history_store[session_id].append({"role": "assistant", "content": bot_response})
    
    except Exception as e:
        return jsonify({"error": f"Error during processing: {e}"}), 500
    
    return jsonify({"response": bot_response})

def get_groq_response(user_message, context_text, session_id):
    try:
        # Truncate the message if it exceeds the maximum length
        if len(context_text) > MAX_MESSAGE_LENGTH - len(user_message) - 50:
            context_text = context_text[:MAX_MESSAGE_LENGTH - len(user_message) - 50] + '...'
        
        # Prepare the chat history for context
        chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history_store[session_id]])
        
        # Add context to the message
        formatted_message = f"User question: {user_message}\nContext: {context_text}\nChat History:\n{chat_history}\nPlease provide a concise and relevant response."
        
        # Get the response from the LLM using the RAG chain
        try:
            chat_completion = rag_chain.invoke({
                "input": formatted_message,
                "chat_history": chat_history_store[session_id],
            })
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"
        
        return chat_completion.get("answer", "I'm sorry, I don't have enough information to answer your question.")
    
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False)