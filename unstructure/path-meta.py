import os
import re
import json

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
    pdf_files = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)

                # Extract building name and tenant name from the directory structure
                parts = root.split(os.sep)
                building_name = parts[-2].replace('_', ' ') if len(parts) >= 2 else ""
                tenant_name = parts[-1].replace('_', ' ') if len(parts) >= 1 else ""

                # Extract date and adjust file name using regex
                match = re.match(r"(\d{4}\.\d{2}\.\d{2}) - (.*)\.pdf", file, re.IGNORECASE)
                if match:
                    date = match.group(1)
                    file_name = match.group(2)
                else:
                    date = ""
                    file_name = file

                pdf_files[file_path] = {
                    "file_name": file_name,
                    "building_name": building_name,
                    "tenant_name": tenant_name,
                    "date": date
                }
    return pdf_files

def save_metadata_to_json(metadata, output_file):
    """
    Saves the metadata to a JSON file.

    Args:
        metadata (dict): The metadata to save.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

if __name__ == "__main__":
    root_folder = "Old_Dominion" 
    output_file = "OldDominion_metadata1.json"

    metadata = extract_pdf_metadata(root_folder)
    save_metadata_to_json(metadata, output_file)

    print(f"Metadata for PDFs has been saved to {output_file}")




# import os
# import re
# import json

# def replace_spaces_with_underscores(dir_path: str):
#     """
#     Replaces spaces with underscores in folder names within the given directory.

#     Args:
#         dir_path (str): The path to the directory to process.
#     """
#     for root, dirs, files in os.walk(dir_path, topdown=False):
#         for dir_name in dirs:
#             if ' ' in dir_name:
#                 old_path = os.path.join(root, dir_name)
#                 new_name = dir_name.replace(' ', '_')
#                 new_path = os.path.join(root, new_name)
#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {old_path} -> {new_path}")

# def extract_pdf_metadata(dir_path: str):
#     """
#     Extracts metadata from PDF files in a given directory and its subdirectories.

#     This function walks through the specified directory and its subdirectories,
#     identifying PDF files and extracting relevant metadata such as file name,
#     building name, tenant name, and date (if available in the file name).

#     Args:
#         dir_path (str): The path to the directory containing PDF files.

#     Returns:
#         dict[str, dict[str, str]]: A dictionary where keys are file paths and
#         values are dictionaries containing extracted metadata for each PDF file.
#     """
#     replace_spaces_with_underscores(dir_path)
#     pdf_files: dict[str, dict[str, str]] = {}
#     for root, dirs, files in os.walk(dir_path):
#         for file in files:
#             if file.lower().endswith('.pdf'):
#                 file_path = os.path.join(root, file)
#                 parts = root.split(os.sep)
#                 building_name = parts[-2] if len(parts) >= 2 else ""

#                 # Extract date and tenant name from file name using regex
#                 match = re.match(
#                     r"(?:(\d{4}\.\d{2}\.\d{2}) - )?(.*?) - (.*)\.pdf", file, re.IGNORECASE)
#                 if match:
#                     date = match.group(1) or ""
#                     tenant_name = match.group(2)
#                     file_name = match.group(3)
#                 else:
#                     date = ""
#                     tenant_name = ""
#                     file_name = os.path.splitext(file)[0]

#                 pdf_files[file_path] = {
#                     "file_name": file_name,
#                     "building_name": building_name,
#                     "tenant_name": tenant_name,
#                     "date": date
#                 }
#     return pdf_files

# def save_metadata_to_json(metadata, output_file):
#     """
#     Saves the metadata to a JSON file.

#     Args:
#         metadata (dict): The metadata to save.
#         output_file (str): The path to the output JSON file.
#     """
#     with open(output_file, 'w') as json_file:
#         json.dump(metadata, json_file, indent=4)

# if __name__ == "__main__":
#     root_folder = "one-folder" 
#     output_file = "pdf_metadata_1.json"

#     metadata = extract_pdf_metadata(root_folder)
#     save_metadata_to_json(metadata, output_file)

#     print(f"Metadata for PDFs has been saved to {output_file}")
