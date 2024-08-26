# import os
# import json
# import fitz  # PyMuPDF
# import re
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from uuid import uuid4
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# import nltk
# nltk.download('averaged_perceptron_tagger')

# load_dotenv()

# jina_api_key = os.getenv("JINAAI_APIKEY") 
# ES_URL = os.getenv("ES_URL")
# ES_API_KEY = os.getenv("ES_API_KEY")

# # Initialize embeddings and vector store
# embeddings = JinaEmbeddings(jina_api_key="jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El", model_name='jina-embeddings-v2-base-en')
# vector_store = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="Old_Dominion_Unstructured",
#     embedding=embeddings,
#     es_api_key=ES_API_KEY
# )


# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             text += page.get_text("text")  # Extract text only
#     # Remove non-alphanumeric characters (except spaces and newlines)
#     text = re.sub(r'[^A-Za-z0-9\s\n]+', '', text)
#     return text

# # Initialize the RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=4000,  # Maximum size of each chunk
#     chunk_overlap=100,  # Overlap between chunks
#     length_function=len,  # Function to determine the length of the text
#     is_separator_regex=False  # Whether to interpret the separator list as regex
# )

# # Load JSON file
# with open('OldDominion_metadata.json', 'r') as file:
#     data = json.load(file)

# # Prepare documents for insertion
# documents = []
# uuids = []

# for pdf_path, metadata in data.items():
#     content = extract_text_from_pdf(pdf_path)
    
#     # Create documents using the RecursiveCharacterTextSplitter
#     chunks = text_splitter.create_documents([content])  # Pass a list containing the content

#     for chunk in chunks:
#         document = Document(
#             page_content=chunk.page_content,  # Use the chunked content
#             metadata={
#                 "file_name": metadata.get("file_name"),
#                 "building_name": metadata.get("building_name"),
#                 "tenant_name": metadata.get("tenant_name"),
#                 "date": metadata.get("date"),
#                 "source": pdf_path  # Include the PDF path in metadata
#             }
#         )
#         documents.append(document)
#         uuids.append(str(uuid4()))  # Generate a unique ID for each document

# # Add documents to the vector store
# vector_store.add_documents(documents=documents, ids=uuids)



# import os
# import json
# from uuid import uuid4
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# import nltk
# from unstructured_client.models import operations, shared
# import unstructured_client

# nltk.download('averaged_perceptron_tagger')

# load_dotenv()

# # Set up environment variables and initialize clients
# jina_api_key = "jina_8cb5dc6a1de64812a166753f9148e820kjsisZ-LPRu7mKuLVbKsoSb2trWU"
# ES_URL = "http://154.38.182.130:9200/"
# ES_API_KEY= "UDNJQlQ1QUJ3V2ZzVTRHWVptS1A6OXZCemhCeTBRUk9vVEd5eElDZlp5dw=="
# # UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
# UNSTRUCTURED_API_URL ="https://api.unstructured.io/general/v0/general"
# UNSTRUCTURED_API_KEY="XWVU7aktwctXNintNWeQrriR0MCIgZ"
# client = unstructured_client.UnstructuredClient(
#     api_key_auth=UNSTRUCTURED_API_KEY,
#     server_url=UNSTRUCTURED_API_URL,
# )

# # Initialize embeddings and vector store
# embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
# vector_store = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="old_dominion",
#     embedding=embeddings,
#     es_api_key=ES_API_KEY
# )

# def extract_and_partition_text(pdf_path):
#     with open(pdf_path, "rb") as f:
#         data = f.read()

#     req = operations.PartitionRequest(
#         partition_parameters=shared.PartitionParameters(
#             files=shared.Files(
#                 content=data,
#                 file_name=os.path.basename(pdf_path)
#             ),
#             strategy=shared.Strategy.AUTO,  
#             languages=['eng'],
#             chunking_strategy="by_title",
#             max_characters=2000,
#             split_pdf_page=True,
#             split_pdf_allow_failed=True,
#             split_pdf_concurrency_level=15
#         ),
#     )

#     try:
#         res = client.general.partition(request=req)
#         element_dicts = [element for element in res.elements]
#         return element_dicts

#     except Exception as e:
#         print(f"Error during partitioning: {e}")
#         return None

# # Load metadata JSON file
# with open('OldDominion_metadata.json', 'r') as file:
#     data = json.load(file)

# # Prepare documents for insertion
# documents = []
# uuids = []

# for pdf_path, metadata in data.items():
#     partitioned_data = extract_and_partition_text(pdf_path)
    
#     if partitioned_data:
#         # Process the partitioned chunks directly
#         for element in partitioned_data:
#             content = element.get('text', '')

#             # Create a document directly from the partitioned chunk
#             document = Document(
#                 page_content=content,
#                 metadata={
#                     "file_name": metadata.get("file_name"),
#                     "building_name": metadata.get("building_name"),
#                     "tenant_name": metadata.get("tenant_name"),
#                     "date": metadata.get("date"),
#                     "source": pdf_path
#                 }
#             )
#             documents.append(document)
#             uuids.append(str(uuid4()))

# # Add documents to the vector store
# vector_store.add_documents(documents=documents, ids=uuids)




import jsonlines
from langchain.schema import Document
from typing import Iterable

def save_docs_to_jsonl(documents: Iterable[Document], file_path: str) -> None:
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.dict())

def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents

from langchain_unstructured import UnstructuredLoader
from unstructured_client import UnstructuredClient

file_paths = list(pdf_files.keys())
client = UnstructuredClient(
    api_key_auth="",
    server="free-api"
)

loader = UnstructuredLoader(
    file_path = file_paths,
    partition_via_api=True,
    client=client,
    strategy="hi_res",
    hi_res_model_name="yolox",
    chunking_strategy="by_title",
    pdf_infer_table_structure=True,
    max_characters=3700,
)

docs = loader.load()

save_docs_to_jsonl(docs, "./docs_beverly.json")
loaded_docs = load_docs_from_jsonl("./docs_beverly.json")

# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(loaded_docs[:2])