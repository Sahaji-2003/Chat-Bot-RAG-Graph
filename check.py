# import os
# import weaviate
# from weaviate.auth import AuthApiKey
# from dotenv import load_dotenv
# import logging

# # Load environment variables from .env file
# load_dotenv()

# # Ensure you have set these environment variables
# wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
# wcd_api_key = os.getenv("WCD_API_KEY")
# jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# # Initialize Weaviate client
# weaviate_client = weaviate.Client(
#     url=wcd_cluster_url,
#     auth_client_secret=AuthApiKey(api_key=wcd_api_key),
#     additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
# )

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Function to check the Weaviate schema
# def check_schema():
#     schema = weaviate_client.schema.get()
#     logging.debug(f"Weaviate schema: {schema}")
#     print("Weaviate Schema:")
#     print(schema)

# # Function to check stored data in Weaviate
# def check_stored_data():
#     query = """
#     {
#         Get {
#             Article4 {
#                 content
#                 _additional {
#                     vector
#                 }
#             }
#         }
#     }
#     """
#     response = weaviate_client.query.raw(query)
#     logging.debug(f"Stored data in Weaviate: {response}")
#     print("Stored Data in Weaviate:")
#     print(response)

# if __name__ == "__main__":
#     check_schema()
#     check_stored_data()



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
jina_ai_api_key = os.getenv("JINAAI_APIKEY")  # Jina AI API key

# Initialize Weaviate client
weaviate_client = weaviate.Client(
    url=wcd_cluster_url,
    auth_client_secret=AuthApiKey(api_key=wcd_api_key),
    additional_headers={'X-Jinaai-Api-Key': jina_ai_api_key}
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Function to check the Weaviate schema
def check_schema():
    schema = weaviate_client.schema.get()
    logging.debug(f"Weaviate schema: {schema}")
    print("Weaviate Schema:")
    print(schema)

# Function to check stored data in Weaviate
def check_stored_data():
    query = """
    {
        Get {
            Testing1 {
                content
                _additional {
                    vector
                }
            }
        }
    }
    """
    response = weaviate_client.query.raw(query)
    logging.debug(f"Stored data in Weaviate: {response}")
    print("Stored Data in Weaviate:")
    print(response)

if __name__ == "__main__":
    check_schema()
    check_stored_data()
