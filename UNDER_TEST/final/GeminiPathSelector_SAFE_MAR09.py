from typing import List, Dict, Optional
import google.generativeai as genai
from dataclasses import dataclass
import json
import re

@dataclass
class PathSelection:
    selected_path: Dict
    confidence_score: float
    reasoning: str
    question_asked: str
    user_response: str

class GeminiPathSelector:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _create_detailed_path_description(self, path: Dict) -> str:
        """Create a detailed description of a path including relationships and node types"""
        description = []
        chain = path['chain']
        for step in chain["path_sequence"]:
            line_info = f" at line {step['line_number']}" if step['line_number'] is not None else ""
            description.append(
                f"{step['from']['type']} '{step['from']['name']}' {step['relationship']} "
                f"{step['to']['type']} '{step['to']['name']}'{line_info}"
            )
        return " -> ".join(description)
    

    #=================================================== ADDED
    
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


    #def get_paths_and_question(self, retriever,function_name: str, max_depth: int) -> Dict:
    def get_paths_and_question(self, retriever, function_name: str, max_depth: int, user_query:str,result_type: str = None) -> Dict:
        """Get paths and generate question without requiring immediate user response"""
        print("\n------------------- PRPL ASSIST INFO [/chat <- Endpoint, get_paths_and_question <- Function, GeminiPathSelector.py  <- File]-----------------------")

        if result_type:
            result_type = result_type.lower()
            #node_type = "ODL" if result_type == "odl" else "CodeEntity"
            if result_type == "odl_file":
                node_type="ODL"
            elif result_type == "odl_object":
                node_type = "Object"
            else:
                node_type="CodeEntity"
        else:
            # Default to CodeEntity if no type provided
            node_type = "CodeEntity"
        print("\n5. Fetching Complete Chains Containing Node",function_name,"of Node Type",node_type," from Neo4j DB")
        chains = retriever.get_complete_chains(
            node_name=function_name,
            max_depth=max_depth,
            node_type=node_type
        )

        if not chains:
            print(f"\nNo paths found for function {function_name}")
            return None

        #============================================================================== DEBUG
        # Print the complete chains before any filtering
        print("\n6. Fetched all Complete Chains from Neo4j DB: ")
        print("\nAll Complete Chains (Before Filtering):")
        retriever.print_complete_chains(chains, function_name)
        #============================================================================== DEBUG

        formatted_chains = self._format_chains(chains)

        # Check if we have fewer than 20 chains
        if len(formatted_chains) < 20:
            print(f"\n7. Found {len(formatted_chains)} paths (less than 20) - skipping grouping step")
            paths = formatted_chains
        else:
            print(f"\n7. Found {len(formatted_chains)} paths - grouping similar paths")
            grouped_chains = self._group_similar_paths(formatted_chains)
            paths = list(grouped_chains.values())[0]

        # grouped_chains = self._group_similar_paths(formatted_chains)
        # paths = list(grouped_chains.values())[0]

        # print("\n7. Grouped the Fetched Paths ")
        #============================================================= PRINT FN
        # Print representative chains
        print("\n8 .Listing All Paths AFter initial Filtering!!!")
        print(f"\nFound {len(paths)} unique path structures")
        for i, chain_wrapper in enumerate(paths, 1):
            print(f"\nPath {i}:")
            retriever.print_complete_chains([chain_wrapper['chain']], function_name)
            # Print the number of similar paths in this group
            # sig = get_path_signature(chain_wrapper['chain'])
            # similar_count = len(path_groups[sig])
            # if similar_count > 1:
            #     print(f"  [This path structure appears {similar_count} times in the complete set]")
        #============================================================= PRINT FN
        print("\n9. Generating Questions for User to Select One Path BAsed on User Interest using Gemini...")
        question = self._generate_multiple_choice_question(paths, function_name,user_query)
        
        #============================================================================== DEBUG
        print("\nGenerated MCQ for Path Selection : ")
        print(f"\n{question}")
        #============================================================================== DEBUG

        return {
            'paths': paths,
            'question': question
        }
    #=================================================== ADDDED

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


    #====================================================================================================
    #====================================================================================================
    #====================================================================================================
    # def _generate_multiple_choice_question(self, paths: List[Dict], target_node: str) -> str:
    #     """Generate a multiple choice question based on the available paths"""
    #     path_descriptions = [
    #         f"Path {i+1} Summary:\n{self._create_detailed_path_description(path)}\n"
    #         f"Key aspects: {path['relevance_explanation']}"
    #         for i, path in enumerate(paths)
    #     ]
        
    #     prompt = f"""You are helping understand a user's requirements regarding '{target_node}'.
    #     You have these different code paths:

    #     {chr(10).join(path_descriptions)}

    #     Create ONE multiple-choice question that will help identify the user's needs.
        
    #     Guidelines:
    #     1. The question should focus on the user's goal or requirement
    #     2. Each option should map to different functional aspects seen in the paths
    #     3. Use simple, non-technical language
    #     4. Provide 3-4 clear options that reflect different purposes/goals
    #     5. Each option should indirectly correspond to different paths
    #     6. Add "Other (please specify)" as the last option
        
    #     Example good question format:
    #     What are you trying to achieve with this setup?
    #     a) Initialize communication with external devices
    #     b) Set up internal configuration parameters
    #     c) Handle error conditions and recovery
    #     d) Other (please specify)
        
    #     Return ONLY the formatted question with options, no additional text."""
        
    #     response = self.model.generate_content(prompt)
    #     return response.text.strip()
    #====================================================================================================
    #====================================================================================================
    #====================================================================================================
    def _generate_multiple_choice_question(self, paths: List[Dict], target_node: str, user_query: str = None) -> str:
        """
        Generate a multiple choice question based on the available paths and user's original query.
        The question aims to clarify user's intent regarding their original query.
        
        Args:
            paths: List of available code paths
            target_node: The automatically selected node from search results
            user_query: The user's original query/requirement (optional)
        """
        # If there's only one path, no need to generate a question
        # if len(paths) <= 1:
        #     return None
        print("========================================")
        print(" --All Available Models --")
        for model in genai.list_models():
            print(model.name)
            
        path_descriptions = [
            f"Path {i+1} Summary:\n{self._create_detailed_path_description(path)}\n"
            f"Key aspects: {path['relevance_explanation']}"
            for i, path in enumerate(paths)
        ]
        
        context = f"""User query: {user_query if user_query else "Not provided"}
        Available paths fetched from DB(Neo4j) based on user query using vector searching techniques:
        {chr(10).join(path_descriptions)}
        """
        
        prompt = f"""{context}

        Create ONE multiple-choice question that will help clarify the user's intent and select the most relevant path.
        
        Guidelines:
        1. The question should directly relate to the user's original query when provided
        2. Focus on understanding the specific aspect or functionality the user wants to explore
        3. Each option should correspond to different functional aspects seen in the paths
        4. Use clear, purpose-focused language that ties back to user goals
        5. Options should help distinguish between the different available paths
        6. Provide 3-4 clear options that reflect different purposes/goals
        7. Each option should indirectly map to different paths based on their functionality
        8. Add "Other (please specify)" as the last option
        
        Example format (adapt based on context):
        In relation to your query about [aspect from user query], which specific aspect are you most interested in understanding?
        a) [First functional aspect relevant to user's query]
        b) [Second functional aspect relevant to user's query]
        c) [Third functional aspect relevant to user's query]
        d) Other (please specify)
        
        Return ONLY the formatted question with options, no additional text."""
        
        print("\n=====================================================================")
        print("\nCreated Prompt for MCQ Creation : ")
        print(prompt)
        print("\n=====================================================================")
        response = self.model.generate_content(prompt)
        return response.text.strip()
    

    def _select_path_based_on_response(
        self, 
        paths: List[Dict], 
        question: str, 
        user_response: str,
        target_node: str,
        user_query: str
    ) -> PathSelection:
        """Select the most appropriate path based on the user's response"""
        print("\n---------------------------- PRPL ASSIST INFO [/select_path <- Endpoint, _select_path_based_on_response() <- Function, GeminiPathSelector.py  <- File]--------------------------------")
        print("\n2. Prompt Creating for Path Selection...")
        path_descriptions = [
            f"Path {i+1}:\n{self._create_detailed_path_description(path)}\n"
            f"Current relevance: {path['relevance_score']}\n"
            f"Context: {path['relevance_explanation']}"
            for i, path in enumerate(paths)
        ]
        
        prompt = f"""Based on these code paths through node '{target_node}':

        {chr(10).join(path_descriptions)}

        User Query: {user_query}
        Multiple choice question asked by AI Agent to verify user needs: {question}
        User's selected option/response for that question: {user_response}

        Select the most relevant path based on the user query.
        Consider:
        1. Which path best matches the functionality implied by the user's query
        2. The existing relevance scores
        3. How well each path aligns with the user's indicated goal
        
        Return your analysis in this JSON format:
        {{
            "selected_path_index": <0-based index of selected path>,
            "confidence_score": <0-10 score of how well this matches user needs>,
            "reasoning": "<explanation of why this path best matches the user's choice>"
        }}
        """
        print("\n3. Created Prompt for Path Selection : ")
        print(prompt)
        response = self.model.generate_content(prompt)
        analysis = self._extract_json_from_response(response.text)
        print("\n4. Response for that Prompt : ")
        print("\n - Seleceted Path : ",analysis['selected_path_index'])
        print("\n - Confidence Score : ",analysis['confidence_score'])
        print("\n - Reasoning : ",analysis['reasoning'])

        # Fix: Ensure selected_path_index is valid and adjust if necessary
        if analysis['selected_path_index'] >= len(paths):
            print("\nWarning: Selected path index out of range, defaulting to first path")
            analysis['selected_path_index'] = 0

        return PathSelection(
            selected_path=paths[analysis['selected_path_index']],
            confidence_score=analysis['confidence_score'],
            reasoning=analysis['reasoning'],
            question_asked=question,
            user_response=user_response
        )
    
    # def select_path_interactively(
    #     self, 
    #     paths: List[Dict], 
    #     target_node: str
    # ) -> PathSelection:
    #     """
    #     Interactively select the most relevant path by asking the user a clarifying question
        
    #     Args:
    #         paths: List of filtered paths with relevance scores
    #         target_node: Name of the target node
            
    #     Returns:
    #         Selected path with selection metadata
    #     """
    #     # Generate multiple choice question
    #     question = self._generate_multiple_choice_question(paths, target_node)
        
    #     # Get user response
    #     print("\nPlease select the option that best describes your goal:")
    #     print(f"\n{question}")
    #     user_response = input("\nYour choice (enter the letter or full answer): ").strip()
        
    #     # Select path based on response
    #     return self._select_path_based_on_response(
    #         paths=paths,
    #         question=question,
    #         user_response=user_response,
    #         target_node=target_node
    #     )


