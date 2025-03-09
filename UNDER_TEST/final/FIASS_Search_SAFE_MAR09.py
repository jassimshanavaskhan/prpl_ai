from typing import List, Dict, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
from dataclasses import dataclass
from enum import Enum

from google.oauth2.service_account import Credentials
import os
from google.generativeai import configure
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
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
# import sys
# sys.path.append(r"D:\pushing")
from VectorStoreManager import VectorStoreManager
vector_store = VectorStoreManager(embedding_model)
vector_store.load_indices(r"vector_stores")


class SearchStrategy(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    ENSEMBLE = "ensemble"

@dataclass
class SearchResult:
    content: str
    metadata: Dict
    score: float
    strategy: str
    type: str  # Added type field

class AdvancedCodeSearch:
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2)
        )
        
    # def preprocess_query(self, query: str) -> Dict:
    #     """
    #     Enhance the query with additional context and extract search parameters
    #     """
    #     query_info = {
    #         'original': query,
    #         'processed': query.lower(),
    #         'type_hints': [],
    #         'components': [],
    #         'priority_terms': []
    #     }
        
    #     # Detect code-specific terms including ODL
    #     code_indicators = {
    #         'function': ['function', 'method', 'routine', 'call'],
    #         'struct': ['struct', 'structure', 'class', 'type'],
    #         'api': ['api', 'interface', 'endpoint'],
    #         'component': ['component', 'module', 'service'],
    #         'odl': ['odl', 'definition', 'object definition', 'parameter', 'validator']
    #     }
        
    #     # Extract potential ODL-specific terms
    #     odl_specific_terms = ['parameter', 'validator', 'enum', 'nested', 'action']
    #     query_info['odl_terms'] = [term for term in odl_specific_terms if term in query.lower()]
        
    #     for type_name, indicators in code_indicators.items():
    #         if any(ind in query.lower() for ind in indicators):
    #             query_info['type_hints'].append(type_name)
                
    #     return query_info



    def preprocess_query(self, query: str) -> Dict:
        """
        Enhance the query with additional context and extract search parameters.
        Now includes enhanced ODL processing with separate object and file handling.
        """
        query_info = {
            'original': query,
            'processed': query.lower(),
            'type_hints': [],
            'components': [],
            'priority_terms': [],
            'odl_focus': None  # Can be 'object', 'file', or None
        }
        
        # Detect code-specific terms including ODL types
        code_indicators = {
            'function': ['function', 'method', 'routine', 'call'],
            'struct': ['struct', 'structure', 'class', 'type'],
            'api': ['api', 'interface', 'endpoint'],
            'component': ['component', 'module', 'service'],
            'odl_object': [
                'odl object', 'odl parameter', 'object definition',
                'parameter definition', 'object validator', 'object parameter',
                'nested object', 'odl nested'
            ],
            'odl_file': [
                'odl file', 'definition file', 'odl definition',
                'odl implementation', 'odl spec', 'odl specification',
                'definition document'
            ]
        }
        
        # Extract ODL-specific information
        odl_terms = {
            'object_terms': [
                'parameter', 'validator', 'enum', 'nested', 'action',
                'attribute', 'default value', 'callback', 'handler',
                'persistent', 'protected', 'read-only', 'unique', 'key'
            ],
            'file_terms': [
                'file', 'implementation', 'spec', 'specification',
                'document', 'definition file', 'event handlers',
                'populate section', 'define section'
            ]
        }
        
        # Process query for ODL-specific focus
        query_lower = query.lower()
        
        # Count matches for object vs file terms
        object_term_matches = sum(1 for term in odl_terms['object_terms'] if term in query_lower)
        file_term_matches = sum(1 for term in odl_terms['file_terms'] if term in query_lower)
        
        # Determine ODL focus based on term matches
        if object_term_matches > file_term_matches:
            query_info['odl_focus'] = 'object'
            query_info['type_hints'].append('odl_object')
        elif file_term_matches > object_term_matches:
            query_info['odl_focus'] = 'file'
            query_info['type_hints'].append('odl_file')
        elif 'odl' in query_lower:
            # If just "ODL" is mentioned without specific terms, include both
            query_info['type_hints'].extend(['odl_object', 'odl_file'])
        
        # Process other code indicators
        for type_name, indicators in code_indicators.items():
            if type_name not in ['odl_object', 'odl_file']:  # Skip ODL types as they're handled above
                if any(ind in query_lower for ind in indicators):
                    query_info['type_hints'].append(type_name)
        
        # Extract additional ODL context
        query_info['odl_context'] = {
            'has_parameter_focus': any(term in query_lower for term in ['parameter', 'param', 'attribute']),
            'has_validator_focus': any(term in query_lower for term in ['validator', 'validation', 'check']),
            'has_event_focus': any(term in query_lower for term in ['event', 'handler', 'callback']),
            'has_nested_focus': any(term in query_lower for term in ['nested', 'child', 'contained']),
            'has_action_focus': 'action' in query_lower,
            'mentioned_attributes': [attr for attr in ['persistent', 'protected', 'read-only', 'unique', 'key'] 
                                  if attr in query_lower]
        }
        
        # Extract potential component names (assuming they're capitalized words)
        import re
        potential_components = re.findall(r'\b[A-Z][a-zA-Z]*\b', query)
        if potential_components:
            query_info['components'].extend(potential_components)
        
        # Identify priority terms (terms in quotes or uppercase)
        priority_terms = re.findall(r'"([^"]+)"|\b([A-Z]+)\b', query)
        query_info['priority_terms'] = [term[0] or term[1] for term in priority_terms]
        
        return query_info

    # def vector_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
    #     """
    #     Perform vector similarity search across different store types
    #     """
    #     results = []
        
    #     # Determine which stores to search based on query hints
    #     store_types = query_info['type_hints'] if query_info['type_hints'] else self.vector_store_manager.vector_stores.keys()
        
    #     # Prioritize ODL search if ODL terms are present
    #     if query_info.get('odl_terms') and 'odl' in self.vector_store_manager.vector_stores:
    #         store_types = ['odl'] + [st for st in store_types if st != 'odl']
        
    #     for store_type in store_types:
    #         if self.vector_store_manager.vector_stores.get(store_type) is not None:
    #             store_results = self.vector_store_manager.search(
    #                 query_info['processed'],
    #                 store_type,
    #                 k=k
    #             )
                
    #             results.extend([
    #                 SearchResult(
    #                     content=res['document'].page_content,
    #                     metadata=res['metadata'],
    #                     score=res['score'],
    #                     strategy='vector'
    #                 ) for res in store_results
    #             ])
        
    #     return sorted(results, key=lambda x: x.score, reverse=True)[:k]

    # def hybrid_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
    #     """
    #     Combine vector search with traditional text-based search
    #     """
    #     # Get vector search results
    #     vector_results = self.vector_search(query_info, k=k*2)
        
    #     # Special handling for ODL content
    #     odl_results = []
    #     non_odl_results = []
        
    #     for res in vector_results:
    #         if res.metadata.get('type') == 'odl':
    #             odl_results.append(res)
    #         else:
    #             non_odl_results.append(res)
        
    #     # Prepare corpus for BM25
    #     corpus = [res.content for res in vector_results]
    #     bm25 = BM25Okapi([doc.split() for doc in corpus])
        
    #     # Get BM25 scores
    #     bm25_scores = bm25.get_scores(query_info['processed'].split())
        
    #     # Combine scores with ODL-awareness
    #     combined_results = []
    #     for idx, (vec_result, bm25_score) in enumerate(zip(vector_results, bm25_scores)):
    #         normalized_vector_score = 1 / (1 + vec_result.score)
    #         normalized_bm25_score = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
            
    #         # Boost ODL results if query contains ODL-specific terms
    #         boost = 1.2 if vec_result.metadata.get('type') == 'odl' and query_info.get('odl_terms') else 1.0
            
    #         combined_score = ((normalized_vector_score + normalized_bm25_score) / 2) * boost
    #         combined_results.append(
    #             SearchResult(
    #                 content=vec_result.content,
    #                 metadata=vec_result.metadata,
    #                 score=combined_score,
    #                 strategy='hybrid'
    #             )
    #         )
            
    #     return sorted(combined_results, key=lambda x: x.score, reverse=True)[:k]

    # def semantic_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
    #     """
    #     Use cross-encoder for semantic search and re-ranking
    #     """
    #     # Get initial candidates from vector search
    #     candidates = self.vector_search(query_info, k=k*3)
        
    #     # Prepare pairs for cross-encoder
    #     pairs = [(query_info['processed'], res.content) for res in candidates]
        
    #     # Get semantic similarity scores
    #     semantic_scores = self.cross_encoder.predict(pairs)
        
    #     # Apply ODL-specific boosting
    #     semantic_results = []
    #     for res, score in zip(candidates, semantic_scores):
    #         # Boost ODL results if query contains ODL-specific terms
    #         boost = 1.2 if res.metadata.get('type') == 'odl' and query_info.get('odl_terms') else 1.0
    #         semantic_results.append(
    #             SearchResult(
    #                 content=res.content,
    #                 metadata=res.metadata,
    #                 score=score * boost,
    #                 strategy='semantic'
    #             )
    #         )
        
    #     return sorted(semantic_results, key=lambda x: x.score, reverse=True)[:k]

    # def ensemble_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
    #     """
    #     Combine results from multiple search strategies using weighted voting
    #     """
    #     vector_results = self.vector_search(query_info, k=k)
    #     hybrid_results = self.hybrid_search(query_info, k=k)
    #     semantic_results = self.semantic_search(query_info, k=k)
        
    #     # Create a scoring map for all unique results
    #     result_scores = {}
        
    #     # Adjust weights based on query content
    #     has_odl_terms = bool(query_info.get('odl_terms'))
    #     weights = {
    #         'vector': 0.25 if has_odl_terms else 0.3,
    #         'hybrid': 0.35 if has_odl_terms else 0.3,
    #         'semantic': 0.4
    #     }
        
    #     for results, strategy in [
    #         (vector_results, 'vector'),
    #         (hybrid_results, 'hybrid'),
    #         (semantic_results, 'semantic')
    #     ]:
    #         for idx, res in enumerate(results):
    #             key = (res.content, str(res.metadata))
    #             position_score = 1.0 - (idx / len(results))
                
    #             if key not in result_scores:
    #                 result_scores[key] = {
    #                     'content': res.content,
    #                     'metadata': res.metadata,
    #                     'weighted_score': 0.0,
    #                     'strategies': set()
    #                 }
                
    #             # Apply ODL boost in ensemble scoring
    #             boost = 1.2 if res.metadata.get('type') == 'odl' and has_odl_terms else 1.0
    #             result_scores[key]['weighted_score'] += position_score * weights[strategy] * boost
    #             result_scores[key]['strategies'].add(strategy)
        
    #     ensemble_results = [
    #         SearchResult(
    #             content=info['content'],
    #             metadata=info['metadata'],
    #             score=info['weighted_score'],
    #             strategy=f"ensemble({','.join(info['strategies'])})"
    #         )
    #         for info in result_scores.values()
    #     ]
        
    #     return sorted(ensemble_results, key=lambda x: x.score, reverse=True)[:k]


    def vector_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Perform vector similarity search across different store types
        """
        results = []
        print("Reached 4 \n")
        #store_types = query_info['type_hints'] if query_info['type_hints'] else self.vector_store_manager.vector_stores.keys()

        print(f"Query info: {query_info}")
        print(f"Available store types: {self.vector_store_manager.vector_stores.keys()}")
        store_types = query_info['type_hints'] if query_info['type_hints'] else list(self.vector_store_manager.vector_stores.keys())
        print(f"Selected store types: {store_types}")
        
        if query_info.get('odl_terms') and 'odl' in self.vector_store_manager.vector_stores:
            store_types = ['odl'] + [st for st in store_types if st != 'odl']
        
        for store_type in store_types:
            if self.vector_store_manager.vector_stores.get(store_type) is not None:
                store_results = self.vector_store_manager.search(
                    query_info['processed'],
                    store_type,
                    k=k
                )
                
                results.extend([
                    SearchResult(
                        content=res['document'].page_content,
                        metadata=res['metadata'],
                        score=res['score'],
                        strategy='vector',
                        type=res['metadata'].get('type', 'unknown')  # Extract type from metadata
                    ) for res in store_results
                ])
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:k]

    def hybrid_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Combine vector search with traditional text-based search
        """
        vector_results = self.vector_search(query_info, k=k*2)
        
        # Check if vector_results is empty to avoid division by zero
        if not vector_results:
            return []
        
        corpus = [res.content for res in vector_results]
        bm25 = BM25Okapi([doc.split() for doc in corpus])
        
        bm25_scores = bm25.get_scores(query_info['processed'].split())
        
        combined_results = []
        for idx, (vec_result, bm25_score) in enumerate(zip(vector_results, bm25_scores)):
            normalized_vector_score = 1 / (1 + vec_result.score)
            normalized_bm25_score = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
            
            boost = 1.2 if vec_result.type == 'odl' and query_info.get('odl_terms') else 1.0
            
            combined_score = ((normalized_vector_score + normalized_bm25_score) / 2) * boost
            combined_results.append(
                SearchResult(
                    content=vec_result.content,
                    metadata=vec_result.metadata,
                    score=combined_score,
                    strategy='hybrid',
                    type=vec_result.type
                )
            )
            
        return sorted(combined_results, key=lambda x: x.score, reverse=True)[:k]

    def semantic_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Use cross-encoder for semantic search and re-ranking
        """
        candidates = self.vector_search(query_info, k=k*3)
        
        # Check if candidates is empty to avoid IndexError
        if not candidates:
            return []

        pairs = [(query_info['processed'], res.content) for res in candidates]
        
        semantic_scores = self.cross_encoder.predict(pairs)
        
        semantic_results = []
        for res, score in zip(candidates, semantic_scores):
            boost = 1.2 if res.type == 'odl' and query_info.get('odl_terms') else 1.0
            semantic_results.append(
                SearchResult(
                    content=res.content,
                    metadata=res.metadata,
                    score=score * boost,
                    strategy='semantic',
                    type=res.type
                )
            )
        
        return sorted(semantic_results, key=lambda x: x.score, reverse=True)[:k]

    def ensemble_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Combine results from multiple search strategies using weighted voting
        """
        vector_results = self.vector_search(query_info, k=k)
        hybrid_results = self.hybrid_search(query_info, k=k)
        semantic_results = self.semantic_search(query_info, k=k)
        
        result_scores = {}
        
        has_odl_terms = bool(query_info.get('odl_terms'))
        weights = {
            'vector': 0.25 if has_odl_terms else 0.3,
            'hybrid': 0.35 if has_odl_terms else 0.3,
            'semantic': 0.4
        }
        
        for results, strategy in [
            (vector_results, 'vector'),
            (hybrid_results, 'hybrid'),
            (semantic_results, 'semantic')
        ]:
            for idx, res in enumerate(results):
                key = (res.content, str(res.metadata))
                position_score = 1.0 - (idx / len(results))
                
                if key not in result_scores:
                    result_scores[key] = {
                        'content': res.content,
                        'metadata': res.metadata,
                        'type': res.type,  # Store type information
                        'weighted_score': 0.0,
                        'strategies': set()
                    }
                
                boost = 1.2 if res.type == 'odl' and has_odl_terms else 1.0
                result_scores[key]['weighted_score'] += position_score * weights[strategy] * boost
                result_scores[key]['strategies'].add(strategy)
        
        ensemble_results = [
            SearchResult(
                content=info['content'],
                metadata=info['metadata'],
                score=info['weighted_score'],
                strategy=f"ensemble({','.join(info['strategies'])})",
                type=info['type']  # Include type in ensemble results
            )
            for info in result_scores.values()
        ]
        
        return sorted(ensemble_results, key=lambda x: x.score, reverse=True)[:k]

    def search(self, query: str, strategy: SearchStrategy = SearchStrategy.ENSEMBLE,
               k: int = 5) -> List[SearchResult]:
        """
        Main search method that orchestrates different search strategies
        """
        print("Reached 2 \n")
        query_info = self.preprocess_query(query)
        print("Reached 3 \n")
        #=================================================== SAFE
        if strategy == SearchStrategy.VECTOR:
            return self.vector_search(query_info, k)
        #=================================================== SAFE
        #==================================================== TEST
        # if strategy == SearchStrategy.VECTOR:
        #     # Log the embedding process
        #     print("Generating query embedding...")
        #     try:
        #         results = self.vector_search(query_info,k)
        #         print(f"Search completed. Found {len(results)} results")
        #         return results
        #     except Exception as e:
        #         print(f"Vector search failed: {str(e)}")
        #         raise
        #====================================================== TEST
        elif strategy == SearchStrategy.HYBRID:
            return self.hybrid_search(query_info, k)
        elif strategy == SearchStrategy.SEMANTIC:
            return self.semantic_search(query_info, k)
        else:  # SearchStrategy.ENSEMBLE
            return self.ensemble_search(query_info, k)
        
