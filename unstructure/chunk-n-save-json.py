from dotenv import load_dotenv
load_dotenv()

import re, os

def replace_spaces_with_underscores(dir_path: str):
    """
    Replaces spaces with underscores in folder names within the given directory.

    Args:
        dir_path (str): The path to the directory to process.
    """
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for dir_name in dirs:
            if ' ' in dir_name:
                old_path = os.path.join(root, dir_name)
                new_name = dir_name.replace(' ', '_')
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


def extract_pdf_metadata(dir_path: str):
    """
    Extracts metadata from PDF files in a given directory and its subdirectories.

    This function walks through the specified directory and its subdirectories,
    identifying PDF files and extracting relevant metadata such as file name,
    building name, tenant name, and date (if available in the file name).

    Args:
        dir_path (str): The path to the directory containing PDF files.

    Returns:
        dict[str, dict[str, str]]: A dictionary where keys are file paths and
        values are dictionaries containing extracted metadata for each PDF file.
    """
    replace_spaces_with_underscores(dir_path)
    pdf_files: dict[str, dict[str, str]] = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                parts = root.split(os.sep)
                if len(parts) >= 2:
                    building_name = parts[-2]
                else:
                    building_name = ""

                # Extract date and tenant name from file name using regex
                match = re.match(
                    r"(?:(\d{4}\.\d{2}\.\d{2}) - )?(.*?) - (.*)\.pdf", file, re.IGNORECASE)
                if match:
                    date = match.group(1) or ""
                    tenant_name = match.group(2)
                    file_name = match.group(3)
                else:
                    date = ""
                    tenant_name = ""
                    file_name = os.path.splitext(file)[0]

                # print(f"TName: ", tenant_name.replace("_", " "))
                # print(f"FNAME: ", file_name)
                # print(f"Date: ", date)
                pdf_files[file_path] = {
                    "file_name": file_name.strip(),
                    "building_name": building_name.strip(),
                    "tenant_name": tenant_name.strip(),
                    "date": date
                }
    return pdf_files

pdf_files = extract_pdf_metadata('Res') or {}

import pprint
pp = pprint.PrettyPrinter()

counterMap = {}

for file in pdf_files.values():
    if file['tenant_name'] in counterMap:
        counterMap[file['tenant_name']]["count"] += 1
        counterMap[file['tenant_name']]["files"].append(file['file_name'])
    else:
        counterMap[file['tenant_name']] = {"count": 1, "files": [file['file_name']]}

print(len(pdf_files))

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
import os

file_paths = list(pdf_files.keys())
#client = UnstructuredClient(
#    api_key_auth=os.getenv("US_API_KEY"),
#    server="free-api"
#)

loader = UnstructuredLoader(
    file_path = file_paths,
    #partition_via_api=True,
    #client=client,
    strategy="hi_res",
    hi_res_model_name="yolox",
    chunking_strategy="by_title",
    pdf_infer_table_structure=True,
    max_characters=3700,
)

docs = loader.load()

save_docs_to_jsonl(docs, "./docs_old_dominion.json")
loaded_docs = load_docs_from_jsonl("./docs_old_dominion.json")

# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(loaded_docs[:2])

loaded_docs = load_docs_from_jsonl("./docs_old_dominion.json")

print("LOADED METADATA: ", loaded_docs[1].metadata['filename'])
