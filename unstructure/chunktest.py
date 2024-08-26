import os, json

import unstructured_client
from unstructured_client.models import operations, shared
from dotenv import load_dotenv

load_dotenv()
client = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

filename = r"C:\Users\Lenovo\Documents\GitHub\sahaji\Chat-Bot-RAG-Graph\database\ondc1.pdf"
with open(filename, "rb") as f:
    data = f.read()

req = operations.PartitionRequest(
    
    partition_parameters=shared.PartitionParameters(
        files=shared.Files(
            content=data,
            file_name=filename,
        ),
        # --- Other partition parameters ---
        # Note: Defining `strategy`, `chunking_strategy`, and `output_format`
        # parameters as strings is accepted, but will not pass strict type checking. It is
        # advised to use the defined enum classes as shown below.
        strategy=shared.Strategy.AUTO,  
        languages=['eng'],
        chunking_strategy="by_title",
        # --- PDF partition parameters ---
        split_pdf_page=True,            # If True, splits the PDF file into smaller chunks of pages.
        split_pdf_allow_failed=True,    # If True, the partitioning continues even if some pages fail.
        split_pdf_concurrency_level=15  # Set the number of concurrent request to the maximum value: 15.
    ),
    
)

try:
    res = client.general.partition(request=req)
    element_dicts = [element for element in res.elements]
    json_elements = json.dumps(element_dicts, indent=2)

    # Print the processed data.
    print(json_elements)

    # Write the processed data to a local file.
    with open("PATH_TO_OUTPUT_FILE", "w") as file:
        file.write(json_elements)
except Exception as e:
    print(e)










# # Before calling the API, replace filename and ensure sdk is installed: "pip install unstructured-client"
# # See https://docs.unstructured.io/api-reference/api-services/sdk for more details

# import unstructured_client
# from unstructured_client.models import operations, shared

# client = unstructured_client.UnstructuredClient(
#     api_key_auth="713aUIrp20VT9fZVGbSMssQw2CW7xi",
#     server_url="https://api.unstructuredapp.io",
# )

# filename = "PATH_TO_FILE"
# with open(filename, "rb") as f:
#     data = f.read()

# req = operations.PartitionRequest(
#     partition_parameters=shared.PartitionParameters(
#         files=shared.Files(
#             content=data,
#             file_name=filename,
#         ),
#         # --- Other partition parameters ---
#         # Note: Defining 'strategy', 'chunking_strategy', and 'output_format'
#         # parameters as strings is accepted, but will not pass strict type checking. It is
#         # advised to use the defined enum classes as shown below.
#         strategy=shared.Strategy.AUTO,  
#         chunking_strategy="by_title",
#         languages=['eng'],
#     ),
# )

# try:
#     res = client.general.partition(request=req)
#     print(res.elements[0])
# except Exception as e:
#     print(e)

