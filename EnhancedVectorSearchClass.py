from VectorStoreManager import VectorStoreManager
from logger import logger
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Tuple
import re
from CodeEntityClass import CodeEntity
from LogEntryClass import LogEntry
from LogAnalyzerClass import LogAnalyzer

class EnhancedVectorSearch:
    def __init__(self,gemini_model, vector_store: VectorStoreManager, entities: Dict[str, CodeEntity]):
        self.gemini_model= gemini_model
        self.vector_store = vector_store
        self.entities = entities
        self.log_analyzer = LogAnalyzer(r"C:\Users\39629\Desktop\prpl_assist_final\testing_final\rdklogs\logs")
        
    def contextual_search(self, query: str) -> Tuple[List[CodeEntity], str]:
        # Step 1: Analyze logs for recent context
        log_entries = self.log_analyzer.parse_log_files()
        logger.info(f"-------------------------------------------")
        logger.info(f"log_entries : {log_entries}")
        logger.info(f"-------------------------------------------")
        # Step 2: Extract relevant functions and APIs from logs
        context_functions = set()
        context_apis = set()
        for entry in log_entries:
            if entry.function_name:
                context_functions.add(entry.function_name)
            context_apis.update(entry.api_calls)
        logger.info(f"-------------------------------------------")
        logger.info(f"context_functions : {context_functions}")
        logger.info(f"-------------------------------------------")
        logger.info(f"-------------------------------------------")
        logger.info(f"context_apis : {context_apis}")
        logger.info(f"-------------------------------------------")
        # Step 3: Generate initial context-aware prompt for LLM
        context_prompt = self._generate_context_prompt(query, log_entries)
        logger.info(f"-------------------------------------------")
        logger.info(f"context_prompt : {context_prompt}")
        logger.info(f"-------------------------------------------")
        initial_response = self._get_llm_response(context_prompt)
        logger.info(f"-------------------------------------------")
        logger.info(f"initial_response : {initial_response}")
        logger.info(f"-------------------------------------------")
        # Step 4: Extract mentioned functions and APIs from LLM response
        mentioned_functions = self._extract_functions_from_response(initial_response)
        logger.info(f"-------------------------------------------")
        logger.info(f"mentioned_functions : {mentioned_functions}")
        logger.info(f"-------------------------------------------")
        mentioned_apis = self._extract_apis_from_response(initial_response)
        logger.info(f"-------------------------------------------")
        logger.info(f"mentioned_apis : {mentioned_apis}")
        logger.info(f"-------------------------------------------")
        # Step 5: Perform enhanced vector search
        search_results = self._perform_enhanced_search(
            query,
            context_functions,
            context_apis,
            mentioned_functions,
            mentioned_apis
        )
        logger.info(f"-------------------------------------------")
        logger.info(f"search_results : {search_results}")
        logger.info(f"-------------------------------------------")
        
        # Step 6: Generate final response with comprehensive context
        final_response = self._generate_final_response(
            query,
            search_results,
            initial_response,
            log_entries
        )
        logger.info(f"-------------------------------------------")
        logger.info(f"final_response : {final_response}")
        logger.info(f"-------------------------------------------")
        
        return search_results, final_response
    
    def _perform_enhanced_search(
        self,
        query: str,
        context_functions: Set[str],
        context_apis: Set[str],
        mentioned_functions: Set[str],
        mentioned_apis: Set[str]
    ) -> List[CodeEntity]:
        # Combine all relevant functions and APIs
        all_functions = context_functions.union(mentioned_functions)
        all_apis = context_apis.union(mentioned_apis)
        
        # Perform targeted searches
        function_results = self.vector_store.search(
            query,
            'function',
            k=5,
            filter_dict={'name': {'$in': list(all_functions)}}
        )
        
        api_results = self.vector_store.search(
            query,
            'api',
            k=5,
            filter_dict={'name': {'$in': list(all_apis)}}
        )
        
        # Get related structures
        struct_results = self._find_related_structures(
            function_results + api_results
        )
        
        # Combine and deduplicate results
        all_results = []
        seen_names = set()
        
        for result in function_results + api_results + struct_results:
            name = result['metadata']['name']
            if name not in seen_names and (entity := self.entities.get(name)):
                all_results.append(entity)
                seen_names.add(name)
                
        return all_results
    
    def _find_related_structures(self, function_results: List[Dict]) -> List[Dict]:
        struct_names = set()
        for result in function_results:
            if entity := self.entities.get(result['metadata']['name']):
                struct_names.update(entity.structs_used)
                
        return self.vector_store.search(
            '',  # Empty query since we're filtering by names
            'struct',
            k=len(struct_names),
            filter_dict={'name': {'$in': list(struct_names)}}
        )
    
    def _generate_final_response(
        self,
        query: str,
        entities: List[CodeEntity],
        initial_response: str,
        log_entries: List[LogEntry]
    ) -> str:
        # Create comprehensive context for final response
        context = {
            'query': query,
            'entities': [entity.to_dict() for entity in entities],
            'initial_analysis': initial_response,
            'recent_logs': [
                {
                    'timestamp': entry.timestamp.isoformat(),
                    'component': entry.component,
                    'function': entry.function_name,
                    'apis': list(entry.api_calls)
                }
                for entry in log_entries[-5:]  # Last 5 relevant logs
            ]
        }
        
        prompt = f"""
        Based on the following comprehensive context:
        1. User Query: {query}
        2. Initial Analysis: {initial_response}
        3. Found {len(entities)} relevant code entities
        4. Recent log activity from components: {', '.join(set(e.component for e in log_entries))}
        
        Please provide a detailed response that:
        1. Explains the relevant code flow and component interactions
        2. Highlights any recent issues or patterns from logs
        3. Includes specific function and API references
        4. Mentions relevant data structures and their usage
        5. Provides RDK-specific context and best practices
        
        Focus on accuracy and practical utility for RDK developers.
        """
        
        return self._get_llm_response(prompt)
    
    def _get_llm_response(self, prompt: str) -> str:
        """Generate response using Google's Gemini model"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "Error generating response"

    def _generate_context_prompt(self, query: str, log_entries: List[LogEntry]) -> str:
        """Generate a context-aware prompt based on recent log entries"""
        # Extract recent activity
        recent_components = set()
        recent_functions = set()
        recent_apis = set()
        
        for entry in log_entries[-10:]:  # Look at last 10 entries
            if entry.component:
                recent_components.add(entry.component)
            if entry.function_name:
                recent_functions.add(entry.function_name)
            recent_apis.update(entry.api_calls)
        
        prompt = f"""
        Analyze this query about the RDK codebase: "{query}"
        
        Recent system context:
        - Active components: {', '.join(recent_components)}
        - Recent functions called: {', '.join(recent_functions)}
        - Recent API calls: {', '.join(recent_apis)}
        
        Please analyze:
        1. Which components are most relevant to this query
        2. What specific functions should be examined
        3. Which API calls might be involved
        4. Any potential error conditions or edge cases
        5. Relevant RDK-specific considerations
        
        Focus on RDK architecture patterns and best practices.
        """
        return prompt
    def _extract_functions_from_response(self, response: str) -> Set[str]:
        """Extract function names from LLM response"""
        # Pattern matches:
        # - CamelCase function names
        # - snake_case function names
        # - Names with component prefixes (e.g., CCSP_, RDK_, etc.)
        function_pattern = re.compile(r'\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*|[a-z]+(?:_[a-z]+)*|(?:CCSP_|RDK_|TR181_|PSM_)[A-Za-z0-9_]+)\b')
        
        # Filter out common words and known non-functions
        common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        functions = set()
        for match in function_pattern.finditer(response):
            func_name = match.group()
            if (len(func_name) > 2 and  # Avoid short abbreviations
                func_name.lower() not in common_words and
                not func_name.endswith('_t')):  # Avoid type definitions
                functions.add(func_name)
        
        return functions
    def _extract_apis_from_response(self, response: str) -> Set[str]:
        """Extract API calls from LLM response"""
        # Pattern matches common RDK/CCSP API prefixes
        api_pattern = re.compile(r'\b(?:CCSP_|RDK_|RBUS_|TR181_|CcspCommon_|DM_|PSM_)[A-Za-z0-9_]+\b')
        return set(api_pattern.findall(response))



