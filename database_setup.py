import os
import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Ensure you have set these environment variables
wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
text_file_path = os.getenv("TEXT_FILE_PATH")  # Path to your local text file
jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# Initialize Weaviate client
weaviate_client = weaviate.Client(
    url=wcd_cluster_url,
    auth_client_secret=AuthApiKey(api_key=wcd_api_key),
    additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the Weaviate schema if it doesn't exist
def create_schema():
    schema = {
        "class": "Article5",
        "vectorizer": "text2vec-jinaai",  # Specify the vectorizer at the top level
        "properties": [
            {
                "name": "content",
                "dataType": ["text"]
            }
        ]
    }
    try:
        weaviate_client.schema.create_class(schema)
    except weaviate.exceptions.UnexpectedStatusCodeError as e:
        if e.status_code == 422:
            logging.info("Class 'Article5' already exists.")
        else:
            raise

# Read content from a local text file
def read_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Store text content in Weaviate
def store_content_in_weaviate(content):
    # Create schema if it doesn't exist
    create_schema()

    # Upload the content to Weaviate
    try:
        weaviate_client.data_object.create({
            "content": content
        }, "Article5")
    except weaviate.exceptions.UnexpectedStatusCodeError as e:
        logging.error(f"Failed to upload file: {e}")

if __name__ == '__main__':
    # Read and store the content from the local text file
    content = read_text_file(text_file_path)
    store_content_in_weaviate(content)
