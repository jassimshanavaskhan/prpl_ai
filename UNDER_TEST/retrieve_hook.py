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
        
    def preprocess_query(self, query: str) -> Dict:
        """
        Enhance the query with additional context and extract search parameters
        """
        query_info = {
            'original': query,
            'processed': query.lower(),
            'type_hints': [],
            'components': [],
            'priority_terms': []
        }
        
        # Detect code-specific terms
        code_indicators = {
            'function': ['function', 'method', 'routine', 'call'],
            'struct': ['struct', 'structure', 'class', 'type'],
            'api': ['api', 'interface', 'endpoint'],
            'component': ['component', 'module', 'service']
        }
        
        for type_name, indicators in code_indicators.items():
            if any(ind in query.lower() for ind in indicators):
                query_info['type_hints'].append(type_name)
        
        return query_info

    def vector_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Perform vector similarity search across different store types
        """
        results = []
        
        # Determine which stores to search based on query hints
        store_types = query_info['type_hints'] if query_info['type_hints'] else self.vector_store_manager.vector_stores.keys()
        
        for store_type in store_types:
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
                    strategy='vector'
                ) for res in store_results
            ])
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:k]

    def hybrid_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Combine vector search with traditional text-based search
        """
        # Get vector search results
        vector_results = self.vector_search(query_info, k=k*2)
        
        # Prepare corpus for BM25
        corpus = [res.content for res in vector_results]
        bm25 = BM25Okapi([doc.split() for doc in corpus])
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(query_info['processed'].split())
        
        # Combine scores (normalized)
        combined_results = []
        for idx, (vec_result, bm25_score) in enumerate(zip(vector_results, bm25_scores)):
            normalized_vector_score = 1 / (1 + vec_result.score)  # Convert distance to similarity
            normalized_bm25_score = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
            
            combined_score = (normalized_vector_score + normalized_bm25_score) / 2
            combined_results.append(
                SearchResult(
                    content=vec_result.content,
                    metadata=vec_result.metadata,
                    score=combined_score,
                    strategy='hybrid'
                )
            )
            
        return sorted(combined_results, key=lambda x: x.score, reverse=True)[:k]

    def semantic_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Use cross-encoder for semantic search and re-ranking
        """
        # Get initial candidates from vector search
        candidates = self.vector_search(query_info, k=k*3)
        
        # Prepare pairs for cross-encoder
        pairs = [(query_info['processed'], res.content) for res in candidates]
        
        # Get semantic similarity scores
        semantic_scores = self.cross_encoder.predict(pairs)
        
        # Create new results with semantic scores
        semantic_results = [
            SearchResult(
                content=res.content,
                metadata=res.metadata,
                score=score,
                strategy='semantic'
            )
            for res, score in zip(candidates, semantic_scores)
        ]
        
        return sorted(semantic_results, key=lambda x: x.score, reverse=True)[:k]

    def ensemble_search(self, query_info: Dict, k: int = 5) -> List[SearchResult]:
        """
        Combine results from multiple search strategies using weighted voting
        """
        # Get results from all strategies
        vector_results = self.vector_search(query_info, k=k)
        hybrid_results = self.hybrid_search(query_info, k=k)
        semantic_results = self.semantic_search(query_info, k=k)
        
        # Create a scoring map for all unique results
        result_scores = {}
        
        # Weight for each strategy
        weights = {
            'vector': 0.3,
            'hybrid': 0.3,
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
                        'weighted_score': 0.0,
                        'strategies': set()
                    }
                
                result_scores[key]['weighted_score'] += position_score * weights[strategy]
                result_scores[key]['strategies'].add(strategy)
        
        # Create final ensemble results
        ensemble_results = [
            SearchResult(
                content=info['content'],
                metadata=info['metadata'],
                score=info['weighted_score'],
                strategy=f"ensemble({','.join(info['strategies'])})"
            )
            for info in result_scores.values()
        ]
        
        return sorted(ensemble_results, key=lambda x: x.score, reverse=True)[:k]

    def search(self, query: str, strategy: SearchStrategy = SearchStrategy.ENSEMBLE,
               k: int = 5) -> List[SearchResult]:
        """
        Main search method that orchestrates different search strategies
        """
        query_info = self.preprocess_query(query)
        
        if strategy == SearchStrategy.VECTOR:
            return self.vector_search(query_info, k)
        elif strategy == SearchStrategy.HYBRID:
            return self.hybrid_search(query_info, k)
        elif strategy == SearchStrategy.SEMANTIC:
            return self.semantic_search(query_info, k)
        else:  # SearchStrategy.ENSEMBLE
            return self.ensemble_search(query_info, k)
        

def main():
    # Initialize
    advanced_search = AdvancedCodeSearch(vector_store)

    # Simple search with default ensemble strategy
    results = advanced_search.search("How dslite works?")

    # Try different strategies
    results_vector = advanced_search.search("find authentication functions", 
                                        strategy=SearchStrategy.VECTOR)
    results_semantic = advanced_search.search("find authentication functions", 
                                            strategy=SearchStrategy.SEMANTIC)

    # Each result includes:
    for result in results:
        print(f"Content: {result.content}")
        print(f"Metadata: {result.metadata}")
        print(f"Score: {result.score}")
        print(f"Strategy: {result.strategy}")