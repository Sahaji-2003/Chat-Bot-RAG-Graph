import os
import json
import fitz  # PyMuPDF
import re
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from uuid import uuid4
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")

# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_semantic",
    embedding=embeddings,
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

text_splitter = SemanticChunker(
    embeddings,
    # max_chunk_size=2000,
    # chunk_overlap=100
)

# Load JSON file
with open('pdf_metadata_1.json', 'r') as file:
    data = json.load(file)

# Prepare documents for insertion
documents = []
uuids = []
print("Starting with chunking..............")
for pdf_path, metadata in data.items():
    content = extract_text_from_pdf(pdf_path)
    chunks = text_splitter.create_documents([content])  # Use the SemanticChunker to create chunks
    for chunk in chunks:
        document = Document(
            page_content=chunk.page_content,  # Use the page_content from the chunked Document
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
    print("done........")

# Add documents to the vector store
vector_store.add_documents(documents=documents, ids=uuids)
        