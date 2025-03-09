# query_handler.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import google.generativeai as genai
import json
import re

@dataclass
class QueryContext:
    relevant_docs: List[Dict]  # Vector store results
    neo4j_entities: List[Dict]  # Related entities from Neo4j
    query_type: str  # 'technical_analysis' or 'general_overview'
    confidence_score: float
    reasoning: str

@dataclass
class QueryResponse:
    response: str
    mermaid_diagram: Optional[str] = None
    context_used: QueryContext = None
    status: str = 'direct_response'  # or 'need_selection' for technical paths

class QueryClassifier:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract and parse JSON from Gemini's response, handling potential markdown formatting"""
        # Remove markdown code block markers if present
        clean_text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL)
        
        # Find the JSON object in the cleaned text
        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")
            
        try:
            # Parse the JSON
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def classify_query(self, query: str) -> Dict:
        """Determine if the query needs technical analysis or general overview."""
        prompt = f"""Analyze this user query about a codebase and classify it:
        Query: "{query}"
        
        Determine if this query requires:
        1. Technical Analysis (detailed code path analysis, specific implementation details, function flows)
        2. General Overview (architectural understanding, high-level explanations, conceptual questions)
        
        Return classification in JSON format:
        {{
            "query_type": "technical_analysis" or "general_overview",
            "confidence": <0-1 score>,
            "reasoning": "<explanation>",
            "suggested_approach": "<how to gather context>"
        }}"""
        
        response = self.model.generate_content(prompt)
        return self._extract_json_from_response(response.text)

class EnhancedQueryHandler:
    def __init__(self, vector_store, neo4j_retriever, gemini_api_key: str):
        self.vector_store = vector_store
        self.neo4j_retriever = neo4j_retriever
        self.classifier = QueryClassifier(gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def _gather_general_context(self, query: str, query_classification: Dict) -> QueryContext:
        """Gather comprehensive context for general overview queries."""
        # Get relevant documents from vector store with higher recall
        vector_results = self.vector_store.similarity_search(
            query,
            k=5,  # Get more results for broader context
            score_threshold=0.3  # Lower threshold for better recall
        )
        
        # Extract key terms from query for Neo4j search
        key_terms = self._extract_key_terms(query)
        neo4j_results = []
        
        # Search Neo4j for relevant entities based on key terms
        with self.neo4j_retriever.driver.session() as session:
            for term in key_terms:
                result = session.run("""
                    MATCH (n)
                    WHERE (n:CodeEntity OR n:ODL) AND 
                          (n.name CONTAINS $term OR n.content CONTAINS $term)
                    RETURN n.name, n.type, n.content, n.component
                    LIMIT 5
                """, term=term)
                neo4j_results.extend(result.data())

        return QueryContext(
            relevant_docs=vector_results,
            neo4j_entities=neo4j_results,
            query_type=query_classification['query_type'],
            confidence_score=query_classification['confidence'],
            reasoning=query_classification['reasoning']
        )

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key technical terms from the query."""
        prompt = f"""Extract key technical terms from this query that would be useful for searching a codebase:
        Query: "{query}"
        Return only a comma-separated list of terms."""
        
        response = self.model.generate_content(prompt)
        return [term.strip() for term in response.text.split(',')]

    def _generate_overview_response(self, query: str, context: QueryContext) -> str:
        """Generate a comprehensive overview response using gathered context."""
        # Prepare context for the model
        docs_context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(context.relevant_docs)
        ])
        
        neo4j_context = "\n\n".join([
            f"Component: {entity['name']}\nType: {entity['type']}\n"
            f"Content: {entity['content']}"
            for entity in context.neo4j_entities
        ])
        
        prompt = f"""Based on this user query: "{query}"
        
        Using the following context from the codebase:
        
        Vector Store Documents:
        {docs_context}
        
        Related Components:
        {neo4j_context}
        
        Provide a comprehensive overview that:
        1. Addresses the user's question directly
        2. Explains relevant architectural concepts
        3. Connects different pieces of information coherently
        4. Provides examples where relevant
        5. Suggests areas for further exploration if applicable
        
        Focus on clarity and completeness while maintaining technical accuracy."""
        
        response = self.model.generate_content(prompt)
        return response.text

    async def handle_query(self, query: str) -> QueryResponse:
        """Main query handling method that routes to appropriate processing."""
        # First, classify the query
        classification = self.classifier.classify_query(query)
        
        if classification['query_type'] == 'technical_analysis':
            # Use existing technical analysis path
            return QueryResponse(
                response=None,
                status='need_selection'
            )
        else:
            # Handle as general overview
            context = self._gather_general_context(query, classification)
            response = self._generate_overview_response(query, context)
            
            return QueryResponse(
                response=response,
                context_used=context,
                status='direct_response'
            )