from pathlib import Path
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
    # def __init__(self, embedding_model: GooglePalmEmbeddings):
    #     self.embedding_model = embedding_model
    #     self.vector_stores = {
    #         'function': None,
    #         'struct': None,
    #         'component': None,
    #         'api': None,
    #         'odl': None  # Added ODL vector store
    #     }
    #     self.text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=200
    #     )
    def __init__(self, embedding_model: GooglePalmEmbeddings):
        self.embedding_model = embedding_model
        self.vector_stores = {
            'function': None,
            'struct': None,
            'component': None,
            'api': None,
            'odl_object': None,  # Changed from 'odl' to 'odl_object'
            'odl_file': None     # Added new store for ODL files
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def get_store_info(self):
        # Add method to return debug info about the store
        return {
            'total_documents': len(self.vector_stores),
            'embedding_model': str(self.embedding_model),
            'is_initialized': hasattr(self, 'store')
        }
    
    def save_indices(self, base_path: str = "vector_stores"):
        """Save all vector store indices with explicit safety settings"""
        os.makedirs(base_path, exist_ok=True)
        for store_type, store in self.vector_stores.items():
            if store is not None:
                store_path = os.path.join(base_path, f"{store_type}_index")
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

    def odl_to_embedding_text(self, odl_object: Dict[str, Any]) -> str:
        """Convert ODL object to text format for embedding"""
        text_parts = [f"Object Name: {odl_object['name']}"]
        
        # Add parameters
        if odl_object['parameters']:
            text_parts.append("Parameters:")
            for param_name, param in odl_object['parameters'].items():
                param_text = [f"- {param_name} ({param.param_type.name})"]
                if param.attributes:
                    param_text.append(f"  Attributes: {', '.join(param.attributes)}")
                if param.default_value:
                    param_text.append(f"  Default: {param.default_value}")
                if param.validators:
                    validator_texts = []
                    for validator in param.validators:
                        if validator['type'] == 'check_enum':
                            validator_texts.append(f"Enum values: {', '.join(validator['values'])}")
                        elif validator['type'] == 'check_is_in':
                            validator_texts.append(f"Must be in: {validator['reference']}")
                    if validator_texts:
                        param_text.append(f"  Validators: {'; '.join(validator_texts)}")
                text_parts.append(" ".join(param_text))
        
        # Add nested objects
        if odl_object['nested_objects']:
            text_parts.append("Nested Objects:")
            for nested_name, nested_obj in odl_object['nested_objects'].items():
                text_parts.append(f"- {nested_name}")
        
        # Add actions
        if odl_object.get('actions'):
            text_parts.append("Actions:")
            for action in odl_object['actions']:
                text_parts.append(f"- On {action['event']} call {action['handler']}")
        
        return "\n".join(text_parts)

    # def create_indices(self, entities: List[CodeEntity], odl_entities: Optional[Dict[str, Any]] = None):
    #     """Create specialized indices for different types of searches"""
    #     grouped_entities = {
    #         'function': [],
    #         'struct': [],
    #         'component': set(),
    #         'api': [],
    #         'odl': []  # Added ODL group
    #     }
    #     #====================================================
    #     # # Process ODL entities if provided
    #     # if odl_entities and 'objects' in odl_entities:
    #     #     print("Entere in ODL")
    #     #     for obj_name, obj_data in odl_entities['objects'].items():
    #     #         grouped_entities['odl'].append(obj_data)
    #     #====================================================
    #     # Process ODL entities if provided
    #     if odl_entities:
    #         for filepath, file_data in odl_entities.items():
    #             if isinstance(file_data, dict) and 'objects' in file_data:
    #                 for obj_name, obj_data in file_data['objects'].items():
    #                     # Add filepath to the object data for reference
    #                     obj_data['filepath'] = file_data.get('filepath', filepath)
    #                     grouped_entities['odl'].append(obj_data)        
    #     #====================================================
        
        
    #     # Process regular code entities
    #     for entity in entities:
    #         if entity.type == 'function':
    #             logger.info(f"---------------------------------------")
    #             logger.info(f"Function name : {entity.name}")
    #             logger.info(f"---------------------------------------")
    #             for i in range(0, len(entity.function_calls)):    
    #                 logger.info(f"Called {entity.function_calls[i].function_name} Function")
    #                 logger.info(f"---> from {entity.function_calls[i].component} Component")
    #                 logger.info(f"---------------")
    #             grouped_entities['function'].append(entity)
    #             if any(call.is_api for call in entity.function_calls):
    #                 grouped_entities['api'].append(entity)
    #         elif entity.type == 'struct':
    #             grouped_entities['struct'].append(entity)
    #         grouped_entities['component'].add(entity.component)
        

    #     # Create vector stores
    #     for store_type, items in grouped_entities.items():
    #         if items:
    #             if store_type == 'component':
    #                 texts = list(items)
    #                 metadatas = [{'component': comp} for comp in items]
    #             # elif store_type == 'odl':
    #             #     texts = [self.odl_to_embedding_text(obj) for obj in items]
    #             #     metadatas = [{
    #             #         'name': obj['name'],
    #             #         'type': 'odl',
    #             #         'file_path': odl_entities.get('filepath', '')
    #             #     } for obj in items]
    #             elif store_type == 'odl':
    #                 texts = [self.odl_to_embedding_text(obj) for obj in items]
    #                 metadatas = [{
    #                     'name': obj['name'],
    #                     'type': 'odl',
    #                     'file_path': obj.get('filepath', '')
    #                 } for obj in items]
    #             else:
    #                 texts = [entity.to_embedding_text() for entity in items]
    #                 metadatas = [{
    #                     'name': entity.name,
    #                     'component': entity.component,
    #                     'file_path': entity.file_path,
    #                     'type': entity.type
    #                 } for entity in items]
                
    #             self.vector_stores[store_type] = FAISS.from_texts(
    #                 texts=texts,
    #                 embedding=self.embedding_model,
    #                 metadatas=metadatas
    #             )
        
    #     # Save indices after creation
    #     self.save_indices()



    ##########################################################################
    
    def odl_file_to_embedding_text(self, filepath: str, file_data: Dict[str, Any]) -> str:
        """Convert ODL file data to text format for embedding"""
        text_parts = [
            f"ODL File: {os.path.basename(filepath)}",
            f"Path: {filepath}",
            f"Component: {self._extract_component_from_path(filepath)}",
            f"Total Objects: {len(file_data.get('objects', {}))}"
        ]
        
        # Add event handlers information
        if file_data.get('event_handlers'):
            text_parts.append("\nEvent Handlers:")
            for handler in file_data['event_handlers']:
                handler_text = f"- {handler.event_type} -> {handler.callback}"
                if handler.filter_expr:
                    handler_text += f" (Filter: {handler.filter_expr})"
                text_parts.append(handler_text)
        
        # Add list of object names
        if file_data.get('objects'):
            text_parts.append("\nDefined Objects:")
            for obj_name in file_data['objects'].keys():
                text_parts.append(f"- {obj_name}")
                
        return "\n".join(text_parts)
    ###########################################################################

    def create_indices(self, entities: List[CodeEntity], odl_entities: Optional[Dict[str, Any]] = None):
        """
        Create specialized vector store indices with comprehensive metadata and processing
        
        Args:
            entities (List[CodeEntity]): List of code entities to index
            odl_entities (Dict[str, Any], optional): Dictionary of ODL entities to index
        """
        # Initialize grouped entities with comprehensive structure
        grouped_entities = {
            'function': [],
            'struct': [],
            'component': set(),
            'api': [],
            'odl_object': [],  # Changed from 'odl' to 'odl_object'
            'odl_file': []     # Added for ODL files
        }
        
        # Process ODL entities with rich metadata
        # if odl_entities:
        #     for filepath, file_data in odl_entities.items():
        #         if isinstance(file_data, dict) and 'objects' in file_data:
        #             for obj_name, obj_data in file_data['objects'].items():
        #                 # Enrich ODL object with additional metadata
        #                 enriched_obj_data = obj_data.copy()
        #                 enriched_obj_data.update({
        #                     'filepath': file_data.get('filepath', filepath),
        #                     'filename': os.path.basename(filepath),
        #                     'component': self._extract_component_from_path(filepath)
        #                 })
        #                 grouped_entities['odl'].append(enriched_obj_data)  
        #         # Process ODL entities with rich metadata
        if odl_entities:
            for filepath, file_data in odl_entities.items():
                if isinstance(file_data, dict):
                    # Process ODL file
                    component = self._extract_component_from_path(filepath)
                    filename = os.path.basename(filepath)
                    grouped_entities['odl_file'].append({
                        'filepath': filepath,
                        'file_data': file_data,
                        'component': component,
                        'name': filename  # Add name field using filename
                    })
                    
                    # Process ODL objects
                    if 'objects' in file_data:
                        for obj_name, obj_data in file_data['objects'].items():
                            # Enrich ODL object with additional metadata
                            enriched_obj_data = obj_data.copy()
                            enriched_obj_data.update({
                                'filepath': filepath,
                                'filename': filename,
                                'component': component
                            })
                            grouped_entities['odl_object'].append(enriched_obj_data)
      

        # Process code entities with detailed categorization
        for entity in entities:
            # Function entity processing
            if entity.type == 'function':
                grouped_entities['function'].append(entity)
                
                # Check for API calls and add to API index
                if any(call.is_api for call in entity.function_calls):
                    grouped_entities['api'].append(entity)
            
            # Struct entity processing
            elif entity.type == 'struct':
                grouped_entities['struct'].append(entity)
            
            # Component tracking
            grouped_entities['component'].add(entity.component)
        
        # Create vector stores for each entity type
        for store_type, items in grouped_entities.items():
            if items:
                # Metadata and text generation strategy per store type
                if store_type == 'component':
                    texts = list(items)
                    metadatas = [{'component': comp, 'type': 'component'} for comp in items]
                
                # elif store_type == 'odl':
                #     texts = [self.odl_to_embedding_text(obj) for obj in items]
                #     metadatas = [{
                #         'name': obj['name'],
                #         'type': 'odl',
                #         'file_path': obj.get('filepath', ''),
                #         'filename': obj.get('filename', ''),
                #         'component': obj.get('component', '')
                #     } for obj in items]
                elif store_type == 'odl_file':
                    texts = [self.odl_file_to_embedding_text(item['filepath'], item['file_data']) 
                            for item in items]
                    metadatas = [{
                        'name': item['name'],  # Include name in metadata
                        'type': 'odl_file',
                        'file_path': item['filepath'],
                        'filename': os.path.basename(item['filepath']),
                        'component': item['component']
                    } for item in items]
                
                elif store_type == 'odl_object':
                    texts = [self.odl_to_embedding_text(obj) for obj in items]
                    metadatas = [{
                        'name': obj['name'],
                        'type': 'odl_object',
                        'file_path': obj.get('filepath', ''),
                        'filename': obj.get('filename', ''),
                        'component': obj.get('component', '')
                    } for obj in items]
                
                
                else:
                    texts = [entity.to_embedding_text() for entity in items]
                    metadatas = [{
                        'name': entity.name,
                        'component': entity.component,
                        'file_path': entity.file_path,
                        'filename': os.path.basename(entity.file_path),
                        'type': entity.type
                    } for entity in items]
                
                # Create FAISS index with comprehensive metadata
                self.vector_stores[store_type] = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_model,
                    metadatas=metadatas
                )
        
        # Save indices after creation
        self.save_indices()

    def _extract_component_from_path(self, filepath: str) -> str:
        """
        Extract component name from filepath
        
        Args:
            filepath (str): Full file path
        
        Returns:
            str: Extracted component name
        """
        # Implement your component extraction logic here
        # This could involve parsing directory structure, filename patterns, etc.
        parts = Path(filepath).parts
        # Example: return a sensible component name based on directory structure
        return parts[-2] if len(parts) > 1 else 'unknown'

    
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
        return formatted_results

  