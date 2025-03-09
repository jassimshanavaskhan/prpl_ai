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

def list_vectors_table(vector_store_manager, store_type: str) -> None:
    """
    Display vectors from a specific store type in a formatted table.
    
    Args:
        vector_store_manager: Instance of VectorStoreManager
        store_type: Type of store to display ('function', 'struct', 'component', 'api', or 'odl')
    """
    if vector_store_manager.vector_stores[store_type] is None:
        print(f"No vectors found in {store_type} store.")
        return
    
    store = vector_store_manager.vector_stores[store_type]
    
    # Get all document IDs and their corresponding documents
    docstore = store.docstore
    
    # Prepare table data
    table_data = []
    for idx, doc_id in enumerate(store.index_to_docstore_id.values()):
        doc = docstore.search(doc_id)
        metadata = doc.metadata
        
        # Create row based on store type
        if store_type == 'component':
            row = [
                idx + 1,
                metadata.get('component', 'N/A'),
                doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            ]
            headers = ['Index', 'Component', 'Content Preview']
        elif store_type == 'odl':
            row = [
                idx + 1,
                metadata.get('name', 'N/A'),
                metadata.get('type', 'odl'),
                metadata.get('file_path', 'N/A'),
                doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            ]
            headers = ['Index', 'Object Name', 'Type', 'File Path', 'Content Preview']
        else:
            row = [
                idx + 1,
                metadata.get('name', 'N/A'),
                metadata.get('component', 'N/A'),
                metadata.get('type', 'N/A'),
                metadata.get('file_path', 'N/A'),
                doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            ]
            headers = ['Index', 'Name', 'Component', 'Type', 'File Path', 'Content Preview']
        
        table_data.append(row)
    
    # Print the table
    print(f"\n{store_type.upper()} Vectors:")
    print(f"Total vectors: {len(table_data)}")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Example usage:
# sys.path.append(r"D:\pushing")
from VectorStoreManager import VectorStoreManager
vector_manager = VectorStoreManager(embedding_model)
vector_manager.load_indices(r"vector_stores")

#List vectors for each store type including ODL
# store_types = ['function', 'struct', 'component', 'api', 'odl_file', 'odl_object','documentation','web_content']  # Added 'odl' to store types
# for store_type in store_types:
#     list_vectors_table(vector_manager, store_type)


list_vectors_table(vector_manager, 'odl_file')