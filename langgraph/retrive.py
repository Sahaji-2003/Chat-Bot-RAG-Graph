import os
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv

load_dotenv()

jina_api_key = os.getenv("JINAAI_APIKEY") 
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

# Now, to retrieve documents based on a query
query_text = "{'file_name': 'ADP - Assignment of Lease', 'building_name': '11107 - 11109 Sunset Hills', 'tenant_name': 'ADP', 'date': '2022.02.26', 'source': 'one-folder\\11107_-_11109_Sunset_Hills\\ADP\\2022.02.26 - ADP - Assignment of Lease.PDF'}"
results = vector_store.similarity_search(query_text, k=2)

# Process and display results

for result in results:
    print(f"Document ID: {result.metadata.get('source')}")
    print(f"File Name: {result.metadata.get('file_name')}")
    print(f"Building Name: {result.metadata.get('building_name')}")
    print(f"Tenant Name: {result.metadata.get('tenant_name')}")
    print(f"Date: {result.metadata.get('date')}")
    # print(f"Content: {result.page_content}")

print(len(results))    