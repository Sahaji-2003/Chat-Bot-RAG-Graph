# import os
# import json
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_community.retrievers import BM25Retriever
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

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Initialize embeddings and vector store
# embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
# vectorstore = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="langchain_index_recursive4000",
#     embedding=embeddings,
#     es_api_key=ES_API_KEY
# )

# # Define metadata fields for self-querying
# metadata_field_info = [
#     AttributeInfo(name="file_name", description="The name of the file", type="string"),
#     AttributeInfo(name="building_name", description="The name of the building", type="string"),
#     AttributeInfo(name="tenant_name", description="The name of the tenant", type="string"),
#     AttributeInfo(name="date", description="The date of the document in YYYY-MM-DD format", type="string or date"),
#     AttributeInfo(name="source", description="The source of the document", type="string"),
# ]

# # Initialize Groq chat model
# chat_llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

# # Create the self-querying retriever
# self_query_retriever = SelfQueryRetriever.from_llm(
#     chat_llm,
#     vectorstore,
#     document_contents='',
#     document_content_description="Collection of real estate documents with attributes such as tenant names, building names, and lease dates.",
#     metadata_field_info=metadata_field_info,
#     enable_limit=True,
#     verbose=True
# )

# # Configure logging


# # Define the maximum length for the message to send to Groq API
# MAX_MESSAGE_LENGTH = 2048  # Adjust this value based on the Groq API limitations

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')

#     try:
#         # Use self-query retriever to get relevant documents
#         doc_results = self_query_retriever.invoke(user_message, k=3)
        
#         # Use BM25 retriever on the results
#         bm25_retriever = BM25Retriever.from_documents(doc_results)
#         results = bm25_retriever.invoke(user_message, k=5)
        
#         # Extract relevant context from results
#         context_text = " ".join([result.page_content for result in results])
#         print(context_text)


#         # Get response from Groq model
#         bot_response = get_groq_response(user_message, context_text)
#     except Exception as e:
  
#         return jsonify({"error": f"Error during processing: {e}"}), 500

#     return jsonify({"response": bot_response})

# def get_groq_response(user_message, context_text):
#     try:
#         # Truncate the message if it exceeds the maximum length
#         if len(context_text) > MAX_MESSAGE_LENGTH - len(user_message) - 50:
#             context_text = context_text[:MAX_MESSAGE_LENGTH - len(user_message) - 50] + '...'

#         # Add context to the message
#         formatted_message = f"User question: {user_message}\nContext: {context_text}\nPlease provide a concise and relevant response."
#         prompt1 = ChatPromptTemplate.from_messages([("system", formatted_message), ("human", user_message)])

#         # Chain the prompt to the chat LLM
#         chain = prompt1 | chat_llm
#         chat_completion = chain.invoke({
#             "role": "human",
#             "content": formatted_message,
#         })

#         return chat_completion.content
    
#     except Exception as e:
#         # Handle the exception and return an error message or handle it as needed
#         return f"An error occurred while processing your request: {str(e)}"


# if __name__ == '__main__':
#     # Run the Flask app
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
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import logging

# Load environment variables
load_dotenv()
jina_api_key = "jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El"
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
vectorstore = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive4000",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)

# Define metadata fields for self-querying
metadata_field_info = [
    AttributeInfo(name="file_name", description="The name of the file", type="string"),
    AttributeInfo(name="building_name", description="The name of the building", type="string"),
    AttributeInfo(name="tenant_name", description="The name of the tenant", type="string"),
    AttributeInfo(name="date", description="The date of the document in YYYY-MM-DD format", type="string"),
    AttributeInfo(name="source", description="The source of the document", type="string"),
]

# Initialize Groq chat model
chat_llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

document_content_description = (
    "This collection contains various documents related to real estate transactions, "
    "including lease agreements, tenant applications, and property management reports. "
    "Key attributes include tenant names, building names, lease dates, and document sources."
)

# Create the self-querying retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    chat_llm,
    vectorstore,
    document_contents='',
    document_content_description=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# Define the maximum length for the message to send to Groq API
MAX_MESSAGE_LENGTH = 2048

# Function to perform step-back query expansion
def step_back_query_expansion(query):
    # Step 1: Generate a higher-level understanding of the query
    high_level_query = chat_llm.invoke(f"This is the description of the content: {document_content_description}\n"
                                      f"- What is the broader concept or main idea behind the query: '{query}'?\n"
                                      f"Consider the context of real estate and document types. Note: Just give the High-level-query")

    # Step 2: Expand the original query with related terms, synonyms, and specific context
    expanded_queries = chat_llm.invoke(f"Expand the following high-level query into multiple queries with related terms, "
                                      f"synonyms, and context-specific words, considering real estate transactions and "
                                      f"document attributes. Each query should capture different aspects or angles: {high_level_query}\n"
                                      f"Note: Provide all the expanded queries in a single text block")

    # Step 3: Validate and refine the expanded queries based on document content description
    refined_queries = chat_llm.invoke(f"Given the content description: {document_content_description}, "
                                     f"refine the expanded queries below to ensure they are relevant and likely to retrieve the most useful documents:\n\n"
                                     f"{expanded_queries.content}\n"
                                     f"Note: just Provide 4 refined queries in a single text block")

    return refined_queries.content

# Function to perform a search with the expanded query
def perform_search(query, limit=5):
    # Perform step-back query expansion
    expanded_query = step_back_query_expansion(query)
    print(f"Expanded Query: {expanded_query}")

    # Perform the search using the expanded query
    results = vectorstore.similarity_search(expanded_query, k=limit, search_params={"similarity": "cosine"})

    return results

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    use_expansion = request.json.get('use_expansion', False) # set this True to use the query expansion technique

    try:
        if use_expansion:
            # Perform search with step-back query expansion
            results = perform_search(user_message)
        else:
            # Use self-query retriever to get relevant documents
            doc_results = self_query_retriever.invoke(user_message, k=3)
            
            # Use BM25 retriever on the results
            bm25_retriever = BM25Retriever.from_documents(doc_results)
            results = bm25_retriever.invoke(user_message, k=5)
        
        # Extract relevant context from results
        context_text = " ".join([result.page_content for result in results])
        print(context_text)

        # Get response from Groq model
        bot_response = get_groq_response(user_message, context_text)
    except Exception as e:
        return jsonify({"error": f"Error during processing: {e}"}), 500

    return jsonify({"response": bot_response})

def get_groq_response(user_message, context_text):
    try:
        # Truncate the message if it exceeds the maximum length
        if len(context_text) > MAX_MESSAGE_LENGTH - len(user_message) - 50:
            context_text = context_text[:MAX_MESSAGE_LENGTH - len(user_message) - 50] + '...'

        # Add context to the message
        formatted_message = f"User question: {user_message}\nContext: {context_text}\nPlease provide a concise and relevant response."
        prompt1 = ChatPromptTemplate.from_messages([("system", formatted_message), ("human", user_message)])

        # Chain the prompt to the chat LLM
        chain = prompt1 | chat_llm
        chat_completion = chain.invoke({
            "role": "human",
            "content": formatted_message,
        })

        return chat_completion.content
    
    except Exception as e:
        # Handle the exception and return an error message or handle it as needed
        return f"An error occurred while processing your request: {str(e)}"

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False)
