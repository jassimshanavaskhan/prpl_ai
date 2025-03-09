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
#=================================== DEBUG
from UNDER_TEST.final.main import display_results
#=================================== DEBUG

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
        print("\n---------------------------- PRPL ASSIST INFO [/analyze_path <- Endpoint, generate_path_analysis() <- Function, app_new2.py  <- File]--------------------------------")
        try:
            print("\n3. Collecting Contents/context from each node of selected path from neo4j DB")
            entities = self._generate_context_from_path(path_selection.selected_path['chain'])
            print("\n4. Collected Context : ")
            entity_contexts = [
                f"Entity: {entity.name}\n"
                f"Type: {entity.type}\n"
                f"Component: {entity.component}\n"
                f"File: {entity.file_path}\n"
                f"Content:\n{entity.content}\n"
                f"{'=' * 50}"
                for entity in entities
            ]
            i=1
            for entity in entities:
                print("\nEntity No : ",i)
                print("\n - Entity Name : ",entity.name)
                print("\n - Entity Type  : ",entity.type)
                print("\n - Component  : ",entity.component)
                print("\n - Content : ", entity.content)
                i=i+1
            # print(entity_contexts)
            print("\n5. Creating Final Prompt for Response Generation...")
            prompt = self._create_analysis_prompt(
                user_query, 
                path_selection, 
                entity_contexts
            )
            print("\n6. Created Prompt : ")
            print(prompt)

            
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

    def analyze_function_path(self, path_selection:PathSelection,function_name: str, query: str, max_depth: int = 7) -> Optional[AnalysisResult]:
        try:
            print("\n---------------------------- PRPL ASSIST INFO [/analyze_path <- Endpoint, analyze_function_path() <- Function, app_new2.py  <- File]--------------------------------")
            # path_selection = self._get_path_selection(function_name, query, max_depth)
            # if not path_selection:
            #     return None
            
            analysis = self.analyzer.generate_path_analysis(path_selection, query)
            print("\n7. Successfully Generated Final Response using Gemini!!!")
            mermaid_diagram = self.visualizer.generate_mermaid(path_selection)
            print("\n8. Created Mermaid Diagram Successfully!!!")
            print("\n9. Returning back Final Response and Mermaid Diagram to UI for Display!!!")
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
    

    @app.route('/analyze_path', methods=['POST'])
    def analyze_path():
        try:
            #print("\n------------------------------------------------------------")
            print("\n---------------------------- PRPL ASSIST INFO [/analyze_path <- Endpoint, analyze_path() <- Function, app_new2.py  <- File]--------------------------------")
            print("\n1. Reached at Flask Server /analyze_path Endpoint!!!")
            path_selection_data  = request.json['path']
            target_node = request.json['target_node']
            user_query = request.json['user_query']
            # print("-->",selected_path[''])
            path_selection = PathSelection(
                selected_path=path_selection_data['selected_path'],
                confidence_score=path_selection_data['confidence_score'],
                reasoning=path_selection_data['reasoning'],
                question_asked=path_selection_data['question_asked'],
                user_response=path_selection_data['user_response']
            )
            # Use the existing path analysis manager to analyze the path
            print("\n2. passing query(TODO), max_depth(TODO), path_selection, function_Name to prompt creation task...")
            analysis_result = path_analysis_manager.analyze_function_path(
                path_selection = path_selection,
                function_name=target_node,
                #query="Analyze the selected code path",  # You might want to store the original query
                query=user_query,
                max_depth=7
            )
            
            if not analysis_result:
                return jsonify({
                    'error': 'Failed to analyze path'
                }), 404
                
            return jsonify({
                'analysis': analysis_result['analysis'],
                'mermaid_diagram': analysis_result['mermaid_diagram']
            })
            
        except Exception as e:
            app.logger.error(f"Error analyzing path: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500



    @app.route('/select_path', methods=['POST'])
    def select_path():
        try:
            question = request.json['question']
            user_response = request.json['response']
            paths = request.json['paths']
            target_node = request.json['target_node']
            #VNOW1
            user_query = request.json['user_query']
            #=================================== DEBUG
            # print("\n---------------------------")
            print("\n---------------------------- PRPL ASSIST INFO [/select_path <- Endpoint, select_path() <- Function, app_new2.py  <- File]--------------------------------")
            print("\n1. User Response containing data reached at Server endpoint /select_path!!")
            print("\n - Question : ",question)
            print("\n - User Response : ", user_response)
            print("\n - Paths : ", paths)
            print("\n - Target Node : ", target_node)
            print("\n - User Query : ",user_query)
            #=================================== DEBUG

            path_selection = config['path_selector']._select_path_based_on_response(
                paths=paths,
                question=question,
                user_response=user_response,
                target_node=target_node
            )
            #=================================== DEBUG
            print("\n5. Path Selected Successfully !! ")

            print("\n---------------------------")
            print("\nSelected Path Details!!")
            print("\nSelected Path : ",path_selection.selected_path)
            print("\nConfidence Score : ", path_selection.confidence_score)
            print("\nReasoning : ", path_selection.reasoning)
            print("\nGenerated Question : ", path_selection.question_asked)
            print("\nUser Response : ", path_selection.user_response)
            #=================================== DEBUG
            print("\nSending back Selected Path, Confidence Score, Reasoning, Entire Path selection also to UI!! ")
            return jsonify({
                'selected_path': path_selection.selected_path,
                'confidence_score': path_selection.confidence_score,
                'reasoning': path_selection.reasoning,
                #====================== ADDED NOW
                'path_selection': path_selection
                #====================== ADDED NOW
            })
        except Exception as e:
            print(str(e))
            return jsonify({'error': str(e)}), 500


    # Modify the chat endpoint to handle the interactive flow
    @app.route('/chat', methods=['POST'])
    def chat():
        print("\n---------------------------- PRPL ASSIST INFO [/chat <- Endpoint, chat() <- Function, app_new2.py  <- File]--------------------------------")
        print("\n1. User Query Reached at Server Endpoint /chat !!!")
        query = request.json['query']
        print("User Query : ",query)
        try:
            search_results = config['advanced_search'].search(
                query, 
                strategy=config['search_strategy']
            )
            #==================================================== DEBUG
            print(f"\n2. Found {len(search_results)} search results")
            # display_results(search_results)
            for idx, result in enumerate(search_results, 1):
                print(f"\nResult {idx}:")
                print(f"Type: {result.type}")  # Added type display
                print(f"Content: {result.content}")
                print(f"Score: {result.score}")
                print(f"Strategy: {result.strategy}")
            #==================================================== DEBUG
            if not search_results:
                return jsonify({
                    'response': "No results found. Please try a different query.",
                    'mermaid_diagram': None,
                    'status': 'no_results'
                }), 200

            # first_result_name = search_results[0].metadata.get('name')
            first_result = search_results[0]
            first_result_name = first_result.metadata.get('name')
            first_result_type = first_result.type.lower() if first_result.type else None
            #==================================================== DEBUG
            print(f"\n3. Selecting Top Entity from Vector Search results : {first_result_name}")
            print(f"\n4. Analyzing function path for: {first_result_name}")
            #==================================================== DEBUG
            # Get paths and generate question first
            paths = config['path_selector'].get_paths_and_question(
                retriever=config['retriever'],
                function_name=first_result_name,
                max_depth=7,
                result_type=first_result_type,
                #VNOW
                user_query=query
            )
            
            if paths:
                print("\n10. Server Reverting Back the \n\n - Initial Filtered Paths, \n - Generated Question, \n - Target Node \n\nto UI")
                return jsonify({
                    'status': 'need_selection',
                    'paths': paths['paths'],
                    'question': paths['question'],
                    'target_node': first_result_name,
                    #VNOW1
                    'user_query' : query
                })
            
            return jsonify({
                'error': 'No valid paths found'
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
        'search_strategy': SearchStrategy.VECTOR,
        'path_selector': GeminiPathSelector(os.environ.get('GEMINI_API_KEY'))
    }
    
    app = create_app(config)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))