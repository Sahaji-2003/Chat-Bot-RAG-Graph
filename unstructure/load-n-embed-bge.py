import jsonlines
from langchain.schema import Document
from typing import Iterable

def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents

loaded_docs = load_docs_from_jsonl("docs_beverly.json")

from langchain_elasticsearch.vectorstores import ElasticsearchStore
# from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

# embeddings = OpenAIEmbeddings(
#     api_key=os.environ["OPENAI_API_KEY"],
#     model="text-embedding-3-large"
# )

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

es_store = ElasticsearchStore.from_documents(
    documents=loaded_docs,
    es_url=os.environ["ES_URL"],
    es_api_key=os.environ["ES_API_KEY"],
    embedding=embeddings,
    index_name="beverly_bge",
)
