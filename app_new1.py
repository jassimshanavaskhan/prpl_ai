from UNDER_TEST.final.CodePathVisualizer import CodePathVisualizer
from UNDER_TEST.final.GeminiPathSelector import GeminiPathSelector
from UNDER_TEST.final.FIASS_Search import *
from UNDER_TEST.final.Neo4jPathRetriever import Neo4jPathRetriever
from dataclasses import dataclass
from typing import List, Dict, Optional, TypedDict
import os
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Data Classes
@dataclass
class EntityContent:
    name: str
    type: str
    content: str
    file_path: str
    component: Optional[str] = None

@dataclass
class PathSelection:
    selected_path: Dict
    confidence_score: float
    reasoning: str
    question_asked: str
    user_response: str

@dataclass
class PathAnalysisResponse:
    summary: str
    detailed_analysis: str
    recommendations: List[str]
    confidence_score: float

class AnalysisResult(TypedDict):
    analysis: str
    mermaid_diagram: str
    path_selection: Optional[PathSelection]

# Core Analysis Classes
class GeminiPathAnalyzer:
    def __init__(self, neo4j_retriever: 'Neo4jPathRetriever', gemini_api_key: str):
        self.neo4j_retriever = neo4j_retriever
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _get_entity_content(self, entity_name: str, entity_type: str) -> Optional[EntityContent]:
        """Retrieve entity content from Neo4j"""
        with self.neo4j_retriever.driver.session() as session:
            # Combined query for both CodeEntity and ODL entities
            query = """
            MATCH (e)
            WHERE (e:CodeEntity OR e:ODL) AND e.name = $name
            RETURN e.content as content,
                   e.file_path as file_path,
                   e.component as component
            """
            result = session.run(query, name=entity_name)
            record = result.single()
            
            if record:
                return EntityContent(
                    name=entity_name,
                    type=entity_type,
                    content=record['content'],
                    file_path=record['file_path'],
                    component=record['component']
                )
            return None

    def _generate_context_from_path(self, path: Dict) -> List[EntityContent]:
        """Generate context by collecting content for all entities in the path"""
        entities = []
        seen = set()
        
        for step in path['path_sequence']:
            for node in [step['from'], step['to']]:
                if (node['name'], node['type']) not in seen:
                    entity = self._get_entity_content(node['name'], node['type'])
                    if entity:
                        entities.append(entity)
                        seen.add((node['name'], node['type']))
        
        return entities

    def generate_path_analysis(self, path_selection: PathSelection, user_query: str) -> str:
        """Generate comprehensive analysis of the selected path using Gemini"""
        try:
            entities = self._generate_context_from_path(path_selection.selected_path['chain'])
            
            entity_contexts = [
                f"Entity: {entity.name}\n"
                f"Type: {entity.type}\n"
                f"Component: {entity.component}\n"
                f"File: {entity.file_path}\n"
                f"Content:\n{entity.content}\n"
                f"{'=' * 50}"
                for entity in entities
            ]
            
            prompt = self._create_analysis_prompt(
                user_query, 
                path_selection, 
                entity_contexts
            )
            
            response = self.model.generate_content(prompt)
            return response.text
                
        except Exception as e:
            return f"Error generating analysis: {str(e)}"

    def _create_analysis_prompt(self, user_query: str, path_selection: PathSelection, entity_contexts: List[str]) -> str:
        return f"""Analyze this code path based on the user's query: "{user_query}"

Context about the selected path:
Question asked: {path_selection.question_asked}
User's response: {path_selection.user_response}
Path confidence score: {path_selection.confidence_score}

Detailed content of all entities in the path:

{chr(10).join(entity_contexts)}

Provide a comprehensive analysis focusing on:
1. How the components interact and depend on each other
2. The flow of data and control through the path
3. Key functionality and purpose of each component
4. Potential areas for attention or improvement
5. How this relates to the user's original query

Give a detailed, prose-based analysis that explains the code path thoroughly."""

class PathAnalysisManager:
    def __init__(self, retriever, gemini_api_key: str):
        self.retriever = retriever
        self.analyzer = GeminiPathAnalyzer(retriever, gemini_api_key)
        self.selector = GeminiPathSelector(gemini_api_key)
        self.visualizer = CodePathVisualizer()

    def analyze_function_path(self, function_name: str, query: str, max_depth: int = 7) -> Optional[AnalysisResult]:
        try:
            path_selection = self._get_path_selection(function_name, query, max_depth)
            if not path_selection:
                return None
            
            analysis = self.analyzer.generate_path_analysis(path_selection, query)
            mermaid_diagram = self.visualizer.generate_mermaid(path_selection)
            
            return {
                "analysis": analysis,
                "mermaid_diagram": mermaid_diagram,
                "path_selection": path_selection
            }
            
        except Exception as e:
            print(f"Error in path analysis: {str(e)}")
            return None

    def _get_path_selection(self, function_name: str, query: str, max_depth: int) -> Optional[PathSelection]:
        chains = self.retriever.get_complete_chains(
            node_name=function_name,
            max_depth=max_depth,
            node_type="CodeEntity"
        )
        
        if not chains:
            return None

        formatted_chains = self._format_chains(chains)
        grouped_chains = self._group_similar_paths(formatted_chains)
        
        return self.selector.select_path_interactively(
            list(grouped_chains.values())[0],
            function_name
        )

    @staticmethod
    def _format_chains(chains: List[Dict]) -> List[Dict]:
        return [{
            'chain': chain,
            'relevance_score': 1.0,
            'relevance_explanation': (
                "A potential path showing relationships and interactions "
                "with this component"
            )
        } for chain in chains]

    @staticmethod
    def _group_similar_paths(chains: List[Dict]) -> Dict[tuple, List[Dict]]:
        def get_path_signature(chain):
            return tuple(
                (step['from']['type'], step['relationship'], step['to']['type'])
                for step in chain['chain']['path_sequence']
            )
        
        groups = {}
        for chain in chains:
            sig = get_path_signature(chain)
            if sig not in groups:
                groups[sig] = []
            groups[sig].append(chain)
        
        return groups

# Flask Application Setup
app = Flask(__name__)
CORS(app)

def create_app(config):
    path_analysis_manager = PathAnalysisManager(
        retriever=config['retriever'],
        gemini_api_key=os.environ.get('GEMINI_API_KEY')
    )
    
    @app.route('/')
    def index():
        return render_template('index.html')
    

    @app.route('/chat', methods=['POST'])
    def chat():
        query = request.json['query']
        try:
            search_results = config['advanced_search'].search(
                query, 
                strategy=config['search_strategy']
            )
            
            if not search_results:
                return jsonify({
                    'response': "No results found. Please try a different query.",
                    'mermaid_diagram': None,
                    'status': 'no_results'
                }), 200

            first_result_name = search_results[0].metadata.get('name')
            result = path_analysis_manager.analyze_function_path(
                function_name=first_result_name,
                query=query
            )
            
            if result:
                return jsonify({
                    'response': result['analysis'],
                    'mermaid_diagram': result['mermaid_diagram']
                })
            else:
                return jsonify({
                    'error': 'Analysis failed'
                }), 404

        except Exception as e:
            app.logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
            return jsonify({
                'response': "An error occurred while processing your request.",
                'mermaid_diagram': None,
                'error': str(e),
                'status': 'error'
            }), 200

    return app

if __name__ == '__main__':
    # Configuration would be moved to a separate config file in practice
    config = {
        'retriever': Neo4jPathRetriever(
            uri=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        ),
        'advanced_search': AdvancedCodeSearch(vector_store),
        'search_strategy': SearchStrategy.VECTOR
    }
    
    app = create_app(config)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))