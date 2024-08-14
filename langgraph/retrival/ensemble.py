import os
import json
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
jina_api_key = "jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El"
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")

# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_metadata",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)

# Initialize different retrievers
bm25_retriever = BM25Retriever.from_documents(vector_store)  # BM25 retriever for keyword-based search

# Create a similarity retriever using the vector store
# similarity_retriever = vector_store.similarity_search_by_vector()  # Use the vector store as a retriever
similarity_retriever = vector_store.as_retriever()
# Create the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, similarity_retriever],
    weights=[0.5, 0.5]  # Adjust weights as needed
)

# Function to perform a search using ensemble retrieval
def perform_ensemble_search(query, limit=5):
    results = ensemble_retriever.invoke(query, k=limit)  # Pass the limit as k
    return results

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your query: ")  # Get query from user
    limit = int(input("Enter the number of documents to retrieve: "))  # Get limit from user
    results = perform_ensemble_search(user_query, limit)

    # Print the results
    for result in results:
        print("Document Content:", result.page_content)
        print("Metadata:", result.metadata)
        print("-" * 40)