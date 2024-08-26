import os
from datetime import datetime
import jsonlines
def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents

loaded_docs = load_docs_from_jsonl("./docs.json")
def extract_file_details(file_path):
    # Extract the file name from the full path
    file_name = os.path.basename(file_path)
    
    # Split the file name by '-' to get the individual parts
    parts = file_name.split('-')
    
    # Determine if the date is present in the file name
    if len(parts) >= 3:
        date_str = parts[0].strip()
        tenant_name = parts[1].strip()
        file_name = '-'.join(parts[2:]).strip('.pdf').strip()
    else:
        tenant_name = parts[0].strip()
        file_name = '-'.join(parts[1:]).strip('.pdf').strip()
        date_str = ""
    
    # Extract the building name from the file path
    building_name = file_path.split('\\')[1]
    
    return {
        'building_name': building_name,
        'tenant_name': tenant_name,
        'date': date_str,
        'filename': file_name
    }

def append_metadata(doc):
    metadata = extract_file_details(doc.metadata['filename'])
    doc.metadata['building_name'] = metadata['building_name']
    doc.metadata['tenant_name'] = metadata['tenant_name']
    doc.metadata['date'] = metadata['date']
    doc.metadata['filename'] = metadata['filename']
    return doc

for i, doc in enumerate(loaded_docs):
    print("DFILE: ", doc.metadata['filename'], "i: ", i)
    with open("temp.md", "+a") as f:
        f.write(f"DFILELE: {doc.metadata['filename']}")
    append_metadata(doc)