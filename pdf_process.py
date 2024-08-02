

import fitz  # PyMuPDF
import os
import subprocess
import sys
import re

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text("text")  # Extract text only
    # Remove non-alphanumeric characters (except spaces and newlines)
    text = re.sub(r'[^A-Za-z0-9\s\n]+', '', text)
    return text

def setup_and_index_pdfs(pdf_files, input_dir='./ragtest/input', root_dir='./ragtest'):
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)

    # Extract text and save to input directory
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        with open(os.path.join(input_dir, os.path.basename(pdf_file).replace('.pdf', '.txt')), 'w', encoding='utf-8') as f:
            f.write(text)

    # Initialize and run the indexing pipeline
    venv_python = os.path.join(os.path.dirname(sys.executable), 'python')
    subprocess.run([venv_python, "-m", "graphrag.index", "--init", "--root", root_dir])
    subprocess.run([venv_python, "-m", "graphrag.index", "--root", root_dir])

if __name__ == '__main__':
    pdf_files = ["database/ondc1.pdf", "database/ondc2.pdf"]  # Replace with your PDF file paths
    setup_and_index_pdfs(pdf_files)


