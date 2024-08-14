import os
import json
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings # or any other LLM you prefer
from langchain_groq import ChatGroq 
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
jina_api_key = os.getenv("JINAAI_APIKEY") 
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embeddings and vector store
embeddings = JinaEmbeddings(jina_api_key="jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El", model_name='jina-embeddings-v2-base-en')
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive4000",
    embedding=embeddings,
    es_api_key=ES_API_KEY
)


document_content_description = (
    "This collection contains various documents related to real estate transactions, "
    "including lease agreements, tenant applications, and property management reports. "
    "Key attributes include tenant names, building names, lease dates, and document sources."
)
# Initialize the language model (LLM)
llm = ChatGroq(temperature=0,api_key=groq_api_key, model="llama-3.1-8b-instant")  # You can adjust the temperature as needed

# Function to perform step-back query expansion
def step_back_query_expansion(query):
    # Step 1: Generate a higher-level understanding of the query
    high_level_query = llm.invoke(f"This is the description of the content: {document_content_description}\n"
                                  f"- What is the broader concept or main idea behind the query: '{query}'?\n"
                                  f"Consider the context of real estate and document types. Note: Just give the High-level-query")

    # Step 2: Expand the original query with related terms, synonyms, and specific context
    expanded_queries = llm.invoke(f"Expand the following high-level query into multiple queries with related terms, "
                                  f"synonyms, and context-specific words, considering real estate transactions and "
                                  f"document attributes. Each query should capture different aspects or angles: {high_level_query}\n"
                                  f"Note: Provide all the expanded queries in a single text block")

    # Step 3: Validate and refine the expanded queries based on document content description
    refined_queries = llm.invoke(f"Given the content description: {document_content_description}, "
                                 f"refine the expanded queries below to ensure they are relevant and likely to retrieve the most useful documents:\n\n"
                                 f"{expanded_queries.content}\n"
                                 f"Note: just Provide 4 refined queries in a single text block")

    return refined_queries.content

# Function to perform a search with the expanded query
def perform_search(query, limit=10):
    # Perform step-back query expansion
    expanded_query = step_back_query_expansion(query)
    print(f"Expanded Query: {expanded_query}")
    
    # Embed the expanded query
    # embedding_vector = embeddings.embed_query(expanded_query)
        # Fetch documents from Elasticsearch
    # documents = vector_store.search(expanded_query, k=limit, search_type="similarity_score_threshold")  # Use the expanded query to fetch relevant documents
    # return documents
    # Initialize the BM25 retriever with the fetched documents
    # bm25_retriever = BM25Retriever.from_documents(documents)  
    # Perform the similarity search using the embedding vector
    # results = vector_store.similarity_search(expanded_query, k=3) # Use the embedding vector
    results= vector_store.similarity_search(expanded_query, k=5, search_params={"similarity": "cosine"})
      # Load documents for BM25 retriever
    
    # Perform the search using the expanded query
    # results = bm25_retriever.invoke(expanded_query, k=1) 

    
    return results
    


query = "Which of the tenants in my portfolio have termination options? Provide a list, including the property address, tenant name, and a brief summary of the option."
results = perform_search(query)

# Print the results
print(len(results))
for result in results:
    # print(result.page_content)
    print(result.metadata)