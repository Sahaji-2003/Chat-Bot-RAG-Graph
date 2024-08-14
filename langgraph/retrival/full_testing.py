from flask import Flask, request, jsonify
import os
import json
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq 
from dotenv import load_dotenv  # Replace with your actual import
from flask_cors import CORS
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load environment variables
jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key="jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El", model_name='jina-embeddings-v2-base-en')
vectorstore = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive4000",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)

metadata_field_info = [
    AttributeInfo(name="file_name", description="The name of the file", type="string"),
    AttributeInfo(name="building_name", description="The name of the building", type="string"),
    AttributeInfo(name="tenant_name", description="The name of the tenant", type="string"),
    AttributeInfo(name="date", description="The date of the document", type="string"),
    AttributeInfo(name="source", description="The source of the document", type="string"),
]

llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")
document_content_description = "This collection contains various documents related to real estate transactions, including lease agreements, tenant applications, and property management reports."

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True
)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Query the self-querying retriever
    doc = self_query_retriever.invoke(user_message, k=3)
    bm25_retriever = BM25Retriever.from_documents(doc)
    results = bm25_retriever.invoke(user_message, k=5)
    
    # Get the top result (if available)
    response_text = "Sorry, I couldn't find an answer."
    if results:
        response_text = results[0].page_content
    
    return jsonify({'response': response_text})

if __name__ == "__main__":
    app.run(port=5000)
