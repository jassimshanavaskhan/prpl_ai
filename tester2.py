from tabulate import tabulate
from typing import Dict, Optional
from google.oauth2.service_account import Credentials
import os
from google.generativeai import configure
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys

if os.path.exists('credentials.json'):
    credentials = Credentials.from_service_account_file('credentials.json')
else:
    # For production, use environment variables
    credentials_info = {
        "type": os.getenv("GOOGLE_TYPE"),
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
        "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
    }
    credentials = Credentials.from_service_account_info(credentials_info)

configure(credentials=credentials)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    credentials=credentials
)
from VectorStoreManager import VectorStoreManager


# Initialize the VectorStoreManager
vector_store_manager = VectorStoreManager(embedding_model)

# Load the saved indices
vector_store_manager.load_indices(r"vector_stores")

# Now search for the specific ODL file in the 'odl_file' vector store
file_name = "dslite_interfacesettings.odl"
search_results = vector_store_manager.search(
    query=file_name,
    store_type="odl_file",
    k=1,  # Get the most relevant result
    filter_dict={"filename": file_name}  # Filter by filename
)

# Check if we got results
if search_results:
    result = search_results[0]
    print(f"Found file with similarity score: {result['score']}")
    print(f"Metadata: {result['metadata']}")
    
    # The document content will be in the document's page_content
    file_content_representation = result['document'].page_content
    print(f"File content representation: {file_content_representation}")
    
    # If you need the original file content, you'll need to open the actual file
    file_path = result['metadata']['file_path']
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            print(f"Original file content: {original_content}")
    else:
        print(f"Original file not found at: {file_path}")
else:
    print(f"No results found for {file_name}")