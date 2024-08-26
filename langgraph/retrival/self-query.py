import os
import json
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-embeddings-v2-base-en')
vectorstore = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive4000",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)

# Define metadata fields for self-querying
metadata_field_info = [
    AttributeInfo(name="file_name", description="The name of the file", type="string"),
    AttributeInfo(name="building_name", description="The name of the building", type="string"),
    AttributeInfo(name="tenant_name", description="The name of the tenant", type="string"),
    AttributeInfo(name="date", description="The date of the document", type="string"),
    AttributeInfo(name="source", description="The source of the document", type="string"),
]


llm = ChatGroq(temperature=0,api_key=groq_api_key, model="llama-3.1-8b-instant")

document_content_description = (
    "This collection contains various documents related to real estate transactions, "
    "including lease agreements, tenant applications, and property management reports. "
    "Key attributes include tenant names, building names, lease dates, and document sources."
)


# document_content_description = "This collection contains various documents related to real estate transactions, including lease agreements, tenant applications, and property management reports."
# Create the self-querying retriever
# self_query_retriever = SelfQueryRetriever.from_llm(
#     llm=llm,
#     vectorstore=vectorstore,
#     document_contents="Document content for retrieval",
#     document_content_description = document_content_description,
#     metadata_field_info=metadata_field_info,
#     enable_limit=True,
#     verbose=True
# )

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True
)

# Example query
query = "Which of the tenants in my portfolio have termination options? Provide a list, including the property address, tenant name, and a brief summary of the option."
doc = self_query_retriever.invoke(query,k=3)
bm25_retriever = BM25Retriever.from_documents(doc)  
results = bm25_retriever.invoke(query, k=5) 
# Print the results
print(len(results))
for result in results:
    # print(result.page_content)
    print(result.metadata)