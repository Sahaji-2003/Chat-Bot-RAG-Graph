# import weaviate
# from weaviate.auth import AuthApiKey
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Ensure you have set these environment variables
# wcd_url = os.getenv("WCD_DEMO_URL")
# wcd_api_key = os.getenv("WCD_DEMO_RO_KEY")

# client = weaviate.Client(
#     url=wcd_url,                                 # Replace with your Weaviate Cloud URL
#     auth_client_secret=AuthApiKey(api_key=wcd_api_key)
# )

# # Reset the schema if it exists
# schema = client.schema.get()
# if any(class_obj['class'] == "Question" for class_obj in schema['classes']):
#     client.schema.delete_class("Question")

# # Define the new schema
# class_obj = {
#     "class": "Question",
#     "vectorizer": "none",  # We will provide vectors ourselves using Jina
#     "properties": [
#         {
#             "name": "content",
#             "dataType": ["text"]
#         },
#         {
#             "name": "embedding",
#             "dataType": ["number[]"]
#         }
#     ]
# }

# client.schema.create_class(class_obj)



# import weaviate
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()
# # Connect to your WCD instance
# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WCD_CLUSTER_URL"),  # Replace with your WCD URL
#     auth_credentials=weaviate.auth.AuthApiKey(
#         api_key=os.getenv("WCD_API_KEY")
#     )  # Replace with your WCD API key
# )

# # Define the collection properties
# properties = [
#     weaviate.Property(name="title", data_type=["text"]),
#     weaviate.Property(name="description", data_type=["text"])
# ]

# # Create the collection
# client.schema.create_class(
#     class_name="Article",
#     properties=properties,
#     vectorizer="text2vec-jinna"
# )



import weaviate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure you have set these environment variables
wcd_cluster_url = os.getenv("WCD_CLUSTER_URL")
wcd_api_key = os.getenv("WCD_API_KEY")

# Connect to your WCD instance
client = weaviate.Client(
    url=wcd_cluster_url,  # Replace with your WCD URL
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=wcd_api_key)  # Replace with your WCD API key
)

# Define the collection properties
properties = [
    {
        "name": "content",
        "dataType": ["text"]
    },
    {
        "name": "embedding",
        "dataType": ["number[]"]
    }
]

# Create the class (collection)
client.schema.create_class({
    "class": "Article2",
    "properties": properties,
    "vectorizer": "none"  # Specify the vectorizer text2vec-jinaai
})
