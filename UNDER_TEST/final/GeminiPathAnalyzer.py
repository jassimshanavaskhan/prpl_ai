
# from CodePathVisualizer import CodePathVisualizer
# from Neo4jPathRetriever import Neo4jPathRetriever
# from typing import Dict, List, Optional
# import google.generativeai as genai
# from dataclasses import dataclass
# from integrator import *

from UNDER_TEST.final.CodePathVisualizer import CodePathVisualizer
from UNDER_TEST.final.CodePathVisualizer import CodePathVisualizer
from UNDER_TEST.final.Neo4jPathRetriever import Neo4jPathRetriever
from typing import Dict, List, Optional
import google.generativeai as genai
from dataclasses import dataclass
from UNDER_TEST.final.integrator import *



@dataclass
class PathSelection:
    selected_path: Dict
    confidence_score: float
    reasoning: str
    question_asked: str
    user_response: str

import json
import re

@dataclass
class EntityContent:
    name: str
    type: str
    content: str
    file_path: str
    component: Optional[str] = None

@dataclass
class PathAnalysisResponse:
    summary: str
    detailed_analysis: str
    recommendations: List[str]
    confidence_score: float

class GeminiPathAnalyzer:
    def __init__(self, neo4j_retriever: Neo4jPathRetriever, gemini_api_key: str):
        self.neo4j_retriever = neo4j_retriever
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract and parse JSON from Gemini's response, handling various formats"""
        # Remove markdown code block markers if present
        clean_text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', response_text, flags=re.DOTALL)
        
        # Try to find JSON object in the cleaned text
        json_match = re.search(r'\{.*\}', clean_text, flags=re.DOTALL)
        if not json_match:
            # If no JSON object found, try to format the response as JSON
            formatted_response = {
                "summary": "Error formatting response",
                "detailed_analysis": clean_text,
                "recommendations": ["Please try the analysis again"],
                "confidence_score": 0
            }
            return formatted_response
            
        try:
            # Parse the JSON object
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # If JSON parsing fails, return formatted error response
            return {
                "summary": "Error parsing response",
                "detailed_analysis": clean_text,
                "recommendations": ["Please try the analysis again"],
                "confidence_score": 0
            }

    def _get_entity_content(self, entity_name: str, entity_type: str) -> Optional[EntityContent]:
        """Retrieve entity content from Neo4j"""
        with self.neo4j_retriever.driver.session() as session:
            query = """
            MATCH (e:CodeEntity {name: $name})
            RETURN e.content as content,
                   e.file_path as file_path,
                   e.component as component
            """
            result = session.run(query, name=entity_name, entity_type=entity_type)
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

    # def _generate_context_from_path(self, path: Dict) -> List[EntityContent]:
    #     """Generate context by collecting content for all entities in the path"""
    #     entities = []
    #     seen = set()  # Track unique entities
        
    #     for step in path['path_sequence']:
    #         # Process 'from' node
    #         if (step['from']['name'], step['from']['type']) not in seen:
    #             entity = self._get_entity_content(
    #                 step['from']['name'],
    #                 step['from']['type']
    #             )
    #             if entity:
    #                 entities.append(entity)
    #                 seen.add((step['from']['name'], step['from']['type']))
            
    #         # Process 'to' node
    #         if (step['to']['name'], step['to']['type']) not in seen:
    #             entity = self._get_entity_content(
    #                 step['to']['name'],
    #                 step['to']['type']
    #             )
    #             if entity:
    #                 entities.append(entity)
    #                 seen.add((step['to']['name'], step['to']['type']))
        
    #     return entities

    def _generate_context_from_path(self, path: Dict) -> List[EntityContent]:
        """
        Generate context by collecting content for all entities in the path, 
        including both Code Entities and ODL Entities
        
        Args:
            path (Dict): The path containing node sequence
        
        Returns:
            List[EntityContent]: List of entities with their content
        """
        entities = []
        seen = set()  # Track unique entities
        
        def get_entity_content(name: str, entity_type: str) -> Optional[EntityContent]:
            """Helper function to retrieve entity content from different sources"""
            # First try Code Entity
            code_entity = self._get_entity_content(name, entity_type)
            if code_entity:
                return code_entity
            
            # If not found in Code Entity, try ODL Entity
            with self.neo4j_retriever.driver.session() as session:
                # Query for ODL Entity
                query = """
                MATCH (o:ODL)
                WHERE o.filename = $name OR o.filepath CONTAINS $name
                RETURN o.content as content, 
                    o.filepath as file_path, 
                    o.component as component
                """
                result = session.run(query, name=name)
                record = result.single()
                
                if record:
                    return EntityContent(
                        name=name,
                        type=entity_type,
                        content=record['content'],
                        file_path=record['file_path'],
                        component=record['component']
                    )
            
            return None
        
        for step in path['path_sequence']:
            # Process 'from' node
            if (step['from']['name'], step['from']['type']) not in seen:
                entity = get_entity_content(
                    step['from']['name'],
                    step['from']['type']
                )
                if entity:
                    entities.append(entity)
                    seen.add((step['from']['name'], step['from']['type']))
            
            # Process 'to' node
            if (step['to']['name'], step['to']['type']) not in seen:
                entity = get_entity_content(
                    step['to']['name'], 
                    step['to']['type']
                )
                if entity:
                    entities.append(entity)
                    seen.add((step['to']['name'], step['to']['type']))
        
        return entities
    #======================================================
#==================================== SAFE
#     def generate_path_analysis(
#         self,
#         path_selection: PathSelection,
#         user_query: str
#     ) -> PathAnalysisResponse:
#         """Generate comprehensive analysis of the selected path using Gemini"""
#         try:
#             # Get content for all entities in the path
#             entities = self._generate_context_from_path(path_selection.selected_path['chain'])
            
#             # Create context for Gemini
#             entity_contexts = []
#             for entity in entities:
#                 context = (
#                     f"Entity: {entity.name}\n"
#                     f"Type: {entity.type}\n"
#                     f"Component: {entity.component}\n"
#                     f"File: {entity.file_path}\n"
#                     f"Content:\n{entity.content}\n"
#                     f"{'=' * 50}"
#                 )
#                 entity_contexts.append(context)
            
#             # Create the prompt for Gemini
#             prompt = f"""Analyze this code path based on the user's query: "{user_query}"

# Please provide your response in the following JSON format only:
# {{
#     "summary": "<Brief overview of the path and its purpose>",
#     "detailed_analysis": "<Very Detailed explanation of how the components interact>",
#     "recommendations": [
#         "<List of recommendations for understanding this code path>"
#     ],
#     "confidence_score": <0-10 score reflecting confidence>
# }}

# Context about the selected path:
# Question asked: {path_selection.question_asked}
# User's response: {path_selection.user_response}
# Confidence score: {path_selection.confidence_score}

# Detailed content of all entities in the path:

# {chr(10).join(entity_contexts)}

# Focus on:
# 1. How the components interact and depend on each other
# 2. The flow of data and control through the path
# 3. Key functionality and purpose of each component
# 4. Potential areas for attention or improvement
# 5. How this relates to the user's original query

# Remember to format your response as valid JSON with the structure shown above."""

#             # Generate response and handle potential formatting issues
#             print("======================================")
#             print("Prompt : ")
#             print(prompt)
#             response = self.model.generate_content(prompt)
#             analysis = self._extract_json_from_response(response.text)
            
#             return PathAnalysisResponse(
#                 summary=analysis['summary'],
#                 detailed_analysis=analysis['detailed_analysis'],
#                 recommendations=analysis['recommendations'],
#                 confidence_score=float(analysis['confidence_score'])
#             )
            
#         except Exception as e:
#             print(f"Error generating analysis: {str(e)}")
#             return PathAnalysisResponse(
#                 summary="Error generating analysis",
#                 detailed_analysis=f"An error occurred: {str(e)}",
#                 recommendations=["Please try the analysis again"],
#                 confidence_score=0.0
#             )

#================================================ SAFE

# =============================================== MODIFIED
    def generate_path_analysis(
        self,
        path_selection: PathSelection,
        user_query: str
    ) -> str:
        """Generate comprehensive analysis of the selected path using Gemini"""
        try:
            # Get content for all entities in the path
            entities = self._generate_context_from_path(path_selection.selected_path['chain'])
            
            # Create context for Gemini
            entity_contexts = []
            for entity in entities:
                context = (
                    f"Entity: {entity.name}\n"
                    f"Type: {entity.type}\n"
                    f"Component: {entity.component}\n"
                    f"File: {entity.file_path}\n"
                    f"Content:\n{entity.content}\n"
                    f"{'=' * 50}"
                )
                entity_contexts.append(context)
            
            # Create the prompt for Gemini
            prompt = f"""Analyze this code path based on the user's query: "{user_query}"

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

            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
                
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
#================================================ MODIFIED
# ========================================= SAFE
# def analyze_function_path_with_gemini(
#     retriever: Neo4jPathRetriever,
#     function_name: str,
#     query: str,
#     max_depth: int = 7
# ) -> Optional[PathAnalysisResponse]:
#     """
#     Complete function path analysis with Gemini-powered detailed analysis
    
#     Args:
#         retriever: Neo4jPathRetriever instance
#         function_name: Name of the function to analyze
#         query: Natural language query about what user is looking for
#         max_depth: Maximum depth for path traversal
        
#     Returns:
#         Detailed path analysis response
#     """
#     try:
#         # First get path selection using existing interactive analysis
#         path_selection = analyze_function_paths_interactive(
#             retriever=retriever,
#             function_name=function_name,
#             query=query,
#             max_depth=max_depth
#         )
#         # print("++++++++++++++++++ Selected Path +++++++++++++++++++++")
#         # print(path_selection)
#         if not path_selection:
#             return None
        
#         # Initialize analyzer and generate detailed analysis
#         analyzer = GeminiPathAnalyzer(
#             neo4j_retriever=retriever,
#             gemini_api_key=os.environ.get('GEMINI_API_KEY')
#         )
        
#         analysis = analyzer.generate_path_analysis(
#             path_selection=path_selection,
#             user_query=query
#         )
        
#         # Print the analysis
#         print("\nDetailed Path Analysis:")
#         print("=" * 50)
#         print(f"\nSummary:")
#         print(analysis.summary)
#         print("\nDetailed Analysis:")
#         print(analysis.detailed_analysis)
#         print("\nRecommendations:")
#         for i, rec in enumerate(analysis.recommendations, 1):
#             print(f"{i}. {rec}")
#         print(f"\nAnalysis Confidence Score: {analysis.confidence_score}/10")
        
#         return analysis
        
#     except Exception as e:
#         print(f"Error in path analysis: {str(e)}")
#         return None
# ========================================= SAFE

# ========================================= MODIFIED
def analyze_function_path_with_gemini(
    retriever: Neo4jPathRetriever,
    function_name: str,
    query: str,
    max_depth: int = 7
) -> Optional[str]:
    """
    Complete function path analysis with Gemini-powered detailed analysis
    
    Args:
        retriever: Neo4jPathRetriever instance
        function_name: Name of the function to analyze
        query: Natural language query about what user is looking for
        max_depth: Maximum depth for path traversal
        
    Returns:
        Detailed path analysis as a string
    """
    try:
        # First get path selection using existing interactive analysis
        path_selection = analyze_function_paths_interactive(
            retriever=retriever,
            function_name=function_name,
            query=query,
            max_depth=max_depth
        )
        #--------------------- RETURN
        # return PathSelection(
        #     selected_path=paths[analysis['selected_path_index']],
        #     confidence_score=analysis['confidence_score'],
        #     reasoning=analysis['reasoning'],
        #     question_asked=question,
        #     user_response=user_response
        # )
        #------------------------------

        if not path_selection:
            return None
        
        # Initialize analyzer and generate detailed analysis
        analyzer = GeminiPathAnalyzer(
            neo4j_retriever=retriever,
            gemini_api_key=os.environ.get('GEMINI_API_KEY')
        )
        
        analysis = analyzer.generate_path_analysis(
            path_selection=path_selection,
            user_query=query
        )
        
        # Generate Mermaid diagram
        visualizer = CodePathVisualizer()
        mermaid_diagram = visualizer.generate_mermaid(path_selection)


        # Print the analysis
        print("\nDetailed Path Analysis:")
        print("=" * 50)
        print(analysis)
        
        print("\nCode Flow Diagram:")
        print("=" * 50)
        
        return {
            "analysis": analysis,
            "mermaid_diagram": mermaid_diagram,
            "path_selection": path_selection
        }
        
    except Exception as e:
        print(f"Error in path analysis: {str(e)}")
        return None
# ========================================= MODIFIED