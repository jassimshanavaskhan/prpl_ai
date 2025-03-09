from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from CodeEntityClass import CodeEntity
import re
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class LogAnalysisResult:
    functions: Set[str]
    structures: Set[str]
    components: Set[str]
    apis: Set[str]
    relevant_log_snippets: List[str]

class EnhancedRDKAssistant:
    def __init__(self, embedding_model, log_vector_store, code_vector_store, llm):
        self.embedding_model = embedding_model
        self.log_vector_store = log_vector_store
        self.code_vector_store = code_vector_store
        self.llm = llm  # Gemini or other LLM client
        
    def create_log_index(self, logs: List[str]):
        """Create vector store index for RDK logs"""
        texts = []
        metadatas = []
        
        for idx, log in enumerate(logs):
            texts.append(log)
            metadatas.append({
                'log_id': f'log_{idx}',
                'timestamp': self._extract_timestamp(log)  # Implement based on your log format
            })
            
        self.log_vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )
    
    def analyze_logs(self, query: str, k: int = 5) -> LogAnalysisResult:
        """Analyze logs to extract relevant code entities"""
        # Search for relevant logs
        relevant_logs = self.log_vector_store.similarity_search_with_score(query, k=k)
        
        # Combine relevant log snippets
        log_snippets = [log.page_content for log, _ in relevant_logs]
        
        # Create a prompt for the LLM to extract code entities
        prompt = f"""
        Analyze these log snippets and extract mentioned code entities. 
        Log snippets:
        {'\n'.join(log_snippets)}
        
        Please identify:
        1. Function names
        2. Structure names
        3. Component names
        4. API calls
        
        Respond in this exact format:
        Functions: [comma-separated list]
        Structures: [comma-separated list]
        Components: [comma-separated list]
        APIs: [comma-separated list]
        """
        
        # Get structured response from LLM
        analysis = self.llm.generate_content(prompt).text
        
        # Parse LLM response
        parsed = self._parse_llm_response(analysis)
        
        return LogAnalysisResult(
            functions=set(parsed.get('Functions', [])),
            structures=set(parsed.get('Structures', [])),
            components=set(parsed.get('Components', [])),
            apis=set(parsed.get('APIs', [])),
            relevant_log_snippets=log_snippets
        )
    
    def _parse_llm_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the structured LLM response"""
        result = {}
        current_category = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                category, items = line.split(':', 1)
                items = [item.strip() for item in items.strip('[]').split(',') if item.strip()]
                result[category] = items
                
        return result
    
    def search_relevant_entities(self, query: str) -> List[CodeEntity]:
        """Enhanced search incorporating log analysis"""
        # First, analyze logs to find relevant code entities
        log_analysis = self.analyze_logs(query)
        
        relevant_entities = set()
        
        # Search based on both the original query and extracted entities
        for entity_type, names in {
            'function': log_analysis.functions,
            'struct': log_analysis.structures,
            'api': log_analysis.apis
        }.items():
            # Search using original query
            query_results = self.code_vector_store.search(query, entity_type, k=2)
            
            # Search for each extracted entity name
            for name in names:
                entity_results = self.code_vector_store.search(name, entity_type, k=1)
                query_results.extend(entity_results)
            
            # Add unique entities
            for result in query_results:
                entity = self.entities[result['metadata']['name']]
                relevant_entities.add(entity)
        
        return list(relevant_entities)
    
    def generate_response_from_entities(self, query: str, relevant_entities: List[CodeEntity], 
                                     log_analysis: Optional[LogAnalysisResult] = None) -> str:
        """Generate enhanced response incorporating log context"""
        # Prepare context from both code entities and logs
        context = []
        
        # Add relevant log snippets if available
        if log_analysis and log_analysis.relevant_log_snippets:
            context.append("Relevant log context:")
            context.extend(log_analysis.relevant_log_snippets[:2])  # Limit to avoid token overflow
            
        # Add code entity information
        context.append("\nRelevant code entities:")
        for entity in relevant_entities:
            context.append(f"\n{entity.type.capitalize()}: {entity.name}")
            context.append(f"Description: {entity.description}")
            context.append(f"Component: {entity.component}")
        
        # Generate response using the enhanced context
        prompt = f"""
        Query: {query}
        
        Context:
        {'\n'.join(context)}
        
        Please provide a detailed response addressing the query using the provided context.
        Include specific references to both the code entities and relevant log patterns where applicable.
        """
        
        response = self.llm.generate_content(prompt).text
        return response