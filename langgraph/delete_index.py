import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
INDEX_NAME = "my-jina-index-test"  # Replace with the name of the index you want to delete

# Initialize the Elasticsearch client
es = Elasticsearch(
    [ES_URL],
    api_key=ES_API_KEY,
)

# Function to delete an index
def delete_index(index_name):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")

# Example usage
if __name__ == "__main__":
    delete_index(INDEX_NAME)