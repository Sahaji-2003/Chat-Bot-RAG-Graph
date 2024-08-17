from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage
import os
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState

from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import JinaEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Initialize Flask app and CORS
flask_app = Flask(__name__)
CORS(flask_app)

# Load environment variables
load_dotenv()
jina_api_key = "jina_49fbcc36861f46159e2250a6970078a7wLbbIieIp9Vxk1z9pQJ3yOmi-6El"
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ES_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embeddings and vector store
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vectorstore = ElasticsearchStore(
    es_url=ES_URL,
    index_name="langchain_index_recursive_bge",
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

document_content_description = (
    "This collection contains various documents related to real estate transactions, "
    "including lease agreements, tenant applications, and property management reports. "
    "Key attributes include tenant names, building names, lease dates, and document sources."
)

llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-70b-versatile")

# Create the self-querying retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Document content for retrieval",
    document_content_description=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# Function to perform step-back query expansion
def step_back_query_expansion(query):
    high_level_query = llm.invoke(f"This is the description of the content: {document_content_description}\n"
                                  f"- What is the broader concept or main idea behind the query: '{query}'?\n"
                                  f"Consider the context of real estate and document types. Note: Just give the High-level-query")

    expanded_queries = llm.invoke(f"Expand the following high-level query into multiple queries with related terms, "
                                  f"synonyms, and context-specific words, considering real estate transactions and "
                                  f"document attributes. Each query should capture different aspects or angles: {high_level_query}\n"
                                  f"Note: Provide all the expanded queries in a single text block")

    refined_queries = llm.invoke(f"Given the content description: {document_content_description}, "
                                 f"refine the expanded queries below to ensure they are relevant and likely to retrieve the most useful documents:\n\n"
                                 f"{expanded_queries.content}\n"
                                 f"Note: just Provide 4 refined queries in a single text block")

    return refined_queries.content

# Function to perform a search with the expanded query
def perform_search(query, limit=5):
    expanded_query = step_back_query_expansion(query)
    results = vectorstore.similarity_search(expanded_query, k=limit, search_params={"similarity": "cosine"})
    return results

def retrieve_documents(query: str, retrieval_method: str):
    if not query.strip():
        return []
    try:
        if retrieval_method == 'self-query':
            documents = self_query_retriever.invoke(query)
            # bm25_retriever = BM25Retriever.from_documents(doc)  
            # documents = bm25_retriever.invoke(query, k=5) 
        else:
            documents = perform_search(query, limit=3)

        print(len(documents))
        for doc in documents:
            print(doc.metadata)

        return documents
    

    except Exception as e:
        print(f"Error in retrieving documents: {e}")
        return []

def call_model(state: MessagesState, retrieval_method: str = 'self-query'):
    messages = state['messages']
    user_query = messages[-1].content
    retrieved_docs = retrieve_documents(user_query, retrieval_method)
    
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = f"Context:\n{context}\n\nUser Query: {user_query} \n Note: Give in concise, accurate, and meaningful manner " 
    
    try:
        response = llm.invoke([HumanMessage(content=full_prompt)])
    except Exception as e:
        print(f"Error in generating response: {e}")
        response = HumanMessage(content="An error occurred while processing your request.")
    
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add nodes for agent (LLM call) and retrieval (search)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("agent", call_model)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Define the conditional edges
workflow.add_conditional_edges("agent", lambda state: "retrieve" if "retrieval" in state else END)
workflow.add_edge("retrieve", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Compile the graph
app = workflow.compile(checkpointer=checkpointer)

# Flask API
@flask_app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    retrieval_method = data.get('retrieval_method', 'self-query')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_message)],"retrieval": retrieval_method},
            config={"configurable": {"thread_id": 42}}
        )
        response_message = final_state["messages"][-1].content
        return jsonify({'response': response_message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    flask_app.run(port=5000)
