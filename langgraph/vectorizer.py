



# import os
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from uuid import uuid4
# from langchain_core.documents import Document
# import json

# from dotenv import load_dotenv

# load_dotenv()

# # Ensure you have set these environment variables
# groq_api_key = os.getenv("GROQ_API_KEY")
# jina_api_key = os.getenv("JINAAI_APIKEY") 
# ES_URL = os.getenv("ES_URL")
# ES_API_KEY = os.getenv("ES_API_KEY")


# embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')

# vector_store = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="langchain_index",
#     embedding=embeddings,
#     es_api_key=ES_API_KEY
# )


# # text = "This is a test document. how are you"
# # query_result = embeddings.embed_query(text)
# # print(query_result)


# document_1 = Document(
#     page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
# )

# document_2 = Document(
#     page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#     metadata={"source": "news"},
# )

# documents = [
#     document_1,
#     document_2
# ]
# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)

# import os
# import json
# import fitz  # PyMuPDF
# import re
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import JinaEmbeddings
# from uuid import uuid4
# from langchain_core.documents import Document
# from dotenv import load_dotenv

# load_dotenv()


# jina_api_key = os.getenv("JINAAI_APIKEY") 
# ES_URL = os.getenv("ES_URL")
# ES_API_KEY = os.getenv("ES_API_KEY")

# # Initialize embeddings and vector store
# embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
# vector_store = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="langchain_index",
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

# # Load JSON file
# with open('pdf_metadata_1.json', 'r') as file:
#     data = json.load(file)

# # Prepare documents for insertion
# documents = []
# uuids = []

# for pdf_path, metadata in data.items():

#     content = extract_text_from_pdf(pdf_path)

#     document = Document(
#         page_content=content,  # Use the extracted content
#         metadata={
#             "file_name": metadata.get("file_name"),
#             "building_name": metadata.get("building_name"),
#             "tenant_name": metadata.get("tenant_name"),
#             "date": metadata.get("date"),
#             "source": pdf_path  # Include the PDF path in metadata
#         }
#     )
#     documents.append(document)
#     uuids.append(str(uuid4()))  # Generate a unique ID for each document

# # Add documents to the vector store
# vector_store.add_documents(documents=documents, ids=uuids)




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

# load_dotenv()

# jina_api_key = os.getenv("JINAAI_APIKEY") 
# ES_URL = os.getenv("ES_URL")
# ES_API_KEY = os.getenv("ES_API_KEY")

# # Initialize embeddings and vector store
# embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
# vector_store = ElasticsearchStore(
#     es_url=ES_URL,
#     index_name="langchain_index_metadata",
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


# import nltk
# nltk.download('punkt')  # Download the necessary data for sentence tokenization
# from nltk.tokenize import sent_tokenize

# def chunk_text_by_sentence(text, chunk_size=2048, overlap=50):
#     """Chunk text by sentences while respecting the maximum chunk size."""
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = ""
    
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "
    
#     # Add the last chunk if it's not empty
#     if current_chunk:
#         chunks.append(current_chunk.strip())
    
#     # Handle overlap
#     if overlap > 0:
#         overlap_chunks = []
#         for i in range(len(chunks)):
#             if i > 0:
#                 overlap_chunk = chunks[i-1][-overlap:] + " " + chunks[i]
#                 overlap_chunks.append(overlap_chunk[:chunk_size])
#             else:
#                 overlap_chunks.append(chunks[i])
#         return overlap_chunks
#     else:
#         return chunks


# # Load JSON file
# with open('pdf_metadata_2.json', 'r') as file:
#     data = json.load(file)

# # Prepare documents for insertion
# documents = []
# uuids = []

# for pdf_path, metadata in data.items():
#     content = extract_text_from_pdf(pdf_path)
#     chunks = chunk_text_by_sentence(content)  # Chunk the extracted content

#     for chunk in chunks:
#         document = Document(
#             page_content=chunk,  # Use the chunked content
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



import os
import json
import fitz  # PyMuPDF
import re
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")

# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key="jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El", model_name='jina-embeddings-v2-base-en')
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive4000",
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