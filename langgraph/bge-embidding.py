
import os
import json
import fitz  # PyMuPDF
import re
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

load_dotenv()

jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")

# Initialize embeddings and vector store


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# embeddings = JinaEmbeddings(jina_api_key="jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El", model_name='jina-embeddings-v2-base-en')
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive_bge",
    embedding=hf,
    es_api_key=ES_API_KEY
)

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text("text")  # Extract text only
    # Remove non-alphanumeric characters (except spaces and newlines)
    text = re.sub(r'[^A-Za-z0-9\s\n]+', '', text)
    return text

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,  # Maximum size of each chunk
    chunk_overlap=100,  # Overlap between chunks
    length_function=len,  # Function to determine the length of the text
    is_separator_regex=False  # Whether to interpret the separator list as regex
)

# Load JSON file
with open('pdf_metadata_2.json', 'r') as file:
    data = json.load(file)

# Prepare documents for insertion
documents = []
uuids = []

for pdf_path, metadata in data.items():
    content = extract_text_from_pdf(pdf_path)
    
    # Create documents using the RecursiveCharacterTextSplitter
    chunks = text_splitter.create_documents([content])  # Pass a list containing the content

    for chunk in chunks:
        document = Document(
            page_content=chunk.page_content,  # Use the chunked content
            metadata={
                "file_name": metadata.get("file_name"),
                "building_name": metadata.get("building_name"),
                "tenant_name": metadata.get("tenant_name"),
                "date": metadata.get("date"),
                "source": pdf_path  # Include the PDF path in metadata
            }
        )
        documents.append(document)
        uuids.append(str(uuid4()))  # Generate a unique ID for each document

# Add documents to the vector store
vector_store.add_documents(documents=documents, ids=uuids)