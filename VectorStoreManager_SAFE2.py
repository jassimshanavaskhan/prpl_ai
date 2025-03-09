from typing import Dict, List, Set, Any, Optional, Tuple
from CodeEntityClass import CodeEntity
from langchain_community.vectorstores import FAISS
from logger import logger
import json
import os
from datetime import datetime
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from LogAnalysisResult import LogAnalysisResult

class VectorStoreManager:
    def __init__(self, embedding_model: GooglePalmEmbeddings):
        self.embedding_model = embedding_model
        self.vector_stores = {
            'function': None,
            'struct': None,
            'component': None,
            'api': None
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def save_indices(self, base_path: str = "vector_stores"):
        """Save all vector store indices with explicit safety settings"""
        os.makedirs(base_path, exist_ok=True)
        for store_type, store in self.vector_stores.items():
            if store is not None:
                store_path = os.path.join(base_path, f"{store_type}_index")
                # store.save_local(store_path, allow_dangerous_deserialization=True)
                store.save_local(store_path)
                
                # Save metadata separately in JSON format
                metadata_path = os.path.join(base_path, f"{store_type}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'store_type': store_type,
                        'total_vectors': len(store.index_to_docstore_id),
                        'embedding_dimension': store.index.d,
                        'creation_timestamp': datetime.now().isoformat()
                    }, f)
    
    def load_indices(self, base_path: str = "vector_stores") -> bool:
        """Load all vector store indices with explicit safety settings"""
        try:
            for store_type in self.vector_stores.keys():
                store_path = os.path.join(base_path, f"{store_type}_index")
                metadata_path = os.path.join(base_path, f"{store_type}_metadata.json")
                
                if os.path.exists(store_path) and os.path.exists(metadata_path):
                    # Load and verify metadata first
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata['store_type'] != store_type:
                            logger.warning(f"Metadata mismatch for {store_type}, skipping...")
                            continue
                    
                    # Load the vector store with explicit safety setting
                    self.vector_stores[store_type] = FAISS.load_local(
                        store_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Successfully loaded {store_type} index with {metadata['total_vectors']} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading indices: {str(e)}")
            return False
    #--------------------------------------------------------------

    def create_indices(self, entities: List[CodeEntity]):
        """Create specialized indices for different types of searches"""
        grouped_entities = {
            'function': [],
            'struct': [],
            'component': set(),
            'api': []
        }
        
        for entity in entities:
            # logger.info(f"enitity.name : {entity.name}, enitity.type : {entity.type}")
            if entity.type == 'function':
                logger.info(f"---------------------------------------")
                logger.info(f"Function name : {entity.name}")
                logger.info(f"---------------------------------------")
                for i in range(0,len(entity.function_calls)):    
                    logger.info(f"Called {entity.function_calls[i].function_name} Function")
                    logger.info(f"---> from {entity.function_calls[i].component} Component")
                    logger.info(f"---------------")
                    # logger.info(f"Function Calls : ")
                grouped_entities['function'].append(entity)
                if any(call.is_api for call in entity.function_calls):
                    grouped_entities['api'].append(entity)
            elif entity.type == 'struct':
                grouped_entities['struct'].append(entity)
            grouped_entities['component'].add(entity.component)
        
        # Create vector stores
        for store_type, items in grouped_entities.items():
            if items:
                if store_type == 'component':
                    texts = list(items)
                    metadatas = [{'component': comp} for comp in items]
                else:
                    texts = [entity.to_embedding_text() for entity in items]
                    metadatas = [{
                        'name': entity.name,
                        'component': entity.component,
                        'file_path': entity.file_path,
                        'type': entity.type
                    } for entity in items]
                # logger.info(r"--------------------------------------")
                # logger.info(f"texts : {texts}")
                # logger.info(r"--------------------------------------")
                # logger.info(f"metadatas : {metadatas}")
                # logger.info(r"--------------------------------------")
                self.vector_stores[store_type] = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_model,
                    metadatas=metadatas
                )
        
        # Save indices after creation
        self.save_indices()
    
    def search(self, query: str, store_type: str, k: int = 5,
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search specific vector store with optional filtering"""
        if self.vector_stores[store_type] is None:
            return []
        
        search_kwargs = {}
        if filter_dict:
            search_kwargs['filter'] = filter_dict
        
        results = self.vector_stores[store_type].similarity_search_with_score(
            query,
            k=k,
            **search_kwargs
        )
        # Convert results to a more usable format
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'document': doc,
                'score': score,
                'metadata': doc.metadata
            })
        # print("------------> JAS21 formatted_results : ",formatted_results)
        return formatted_results

  