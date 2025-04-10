# # For requests/urllib3
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# import ssl

# # Create an unverified SSL context
# ssl_context = ssl._create_unverified_context()
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
#=========================================================================
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
from logger import logger
#=================================== DEBUG
#=================================== NEW
from flask import jsonify, request
import os
from werkzeug.utils import secure_filename
import shutil
from RDKAssistant_Class import RDKAssistant
from pathlib import Path
from neo4j_Class import Neo4jCodeEntityProcessor
#=================================== NEW


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


# class GeminiResultSelector:
#     def __init__(self, gemini_api_key):
#         self.gemini_api_key = gemini_api_key
#         self.genai = genai.GenerativeModel("gemini-1.5-pro")
#         genai.configure(api_key=self.gemini_api_key)
        
#     def select_best_result(self, search_results, user_query):
#         """
#         Select the most relevant search result based on the user query.
        
#         Args:
#             search_results: List of search results
#             user_query: Original user query
            
#         Returns:
#             best_result: The most relevant search result
#         """
#         if not search_results:
#             return None
            
#         if len(search_results) == 1:
#             return search_results[0]
            
#         # Format results for evaluation, ensuring all values are JSON serializable
#         candidate_results = []
#         for idx, result in enumerate(search_results[:5]):  # Limit to top 5 results
#             # Convert float32 to standard Python float for JSON serialization
#             score = float(result.score) if hasattr(result.score, 'item') else result.score
            
#             result_info = {
#                 "index": idx,
#                 "name": result.metadata.get('name', 'Unknown'),
#                 "type": result.type,
#                 "content": result.content[:1000],  # Truncate long content
#                 "score": score  # Now a standard Python float
#             }
#             candidate_results.append(result_info)
        
#         # Create prompt for Gemini
#         prompt = f"""
#         Given the following user query and search results, select the most relevant result.
        
#         User Query: "{user_query}"
        
#         Search Results:
#         """
        
#         # Add each result individually to avoid JSON serialization issues
#         for idx, result in enumerate(candidate_results):
#             prompt += f"\nResult {idx}:\n"
#             prompt += f"  Name: {result['name']}\n"
#             prompt += f"  Type: {result['type']}\n"
#             prompt += f"  Content: {result['content']}\n"
#             prompt += f"  Score: {result['score']}\n"
        
#         prompt += """
#         Evaluate each result based on:
#         1. Relevance to the user query
#         2. Type of entity (function, struct, ODL object, etc.) that would best answer the query
#         3. Content quality and specificity
        
#         Respond with just the index number of the best result and a brief explanation why.
#         Format: {
#             "selected_index": <index>,
#             "reasoning": "<your reasoning>"
#         }
#         """
        
#         try:
#             response = self.genai.generate_content(prompt)
#             response_text = response.text
            
#             # Extract JSON response
#             match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if match:
#                 result_json = json.loads(match.group(0))
#                 selected_index = result_json.get("selected_index")
                
#                 if selected_index is not None and 0 <= selected_index < len(search_results):
#                     logger.info(f"Gemini selected result {selected_index}: {search_results[selected_index].metadata.get('name')} with reasoning: {result_json.get('reasoning')}")
#                     return search_results[selected_index]
                    
#             # Fallback to highest scored result if parsing fails
#             return max(search_results, key=lambda x: float(x.score) if hasattr(x.score, 'item') else x.score)
            
#         except Exception as e:
#             logger.error(f"Error using Gemini to select result: {str(e)}")
#             # Fallback to highest scored result, ensuring proper conversion
#             return max(search_results, key=lambda x: float(x.score) if hasattr(x.score, 'item') else x.score)
        


# class AdaptiveSearchSelector:
#     def __init__(self, gemini_api_key):
#         self.gemini_api_key = gemini_api_key
#         self.genai = genai.GenerativeModel("gemini-1.5-pro")
#         genai.configure(api_key=self.gemini_api_key)
        
#     def select_best_result(self, advanced_search, user_query, initial_strategy=SearchStrategy.VECTOR):
#         """
#         Adaptively selects the best search result, trying different strategies if needed.
        
#         Args:
#             advanced_search: AdvancedCodeSearch instance
#             user_query: Original user query
#             initial_strategy: Search strategy to try first
            
#         Returns:
#             best_result: The most relevant search result
#             used_strategy: The strategy that produced the best result
#         """
#         # List of strategies to try in order if initial results aren't good
#         strategies = [
#             initial_strategy,
#             SearchStrategy.HYBRID,
#             SearchStrategy.SEMANTIC,
#             SearchStrategy.ENSEMBLE
#         ]
        
#         # Make sure we don't try the same strategy twice
#         strategies = list(dict.fromkeys(strategies))
        
#         best_result = None
#         used_strategy = None
#         attempted_strategies = []
        
#         for strategy in strategies:
#             logger.info(f"Trying search strategy: {strategy.name}")
#             attempted_strategies.append(strategy.name)
            
#             # Perform search with current strategy
#             search_results = advanced_search.search(user_query, strategy=strategy)

#             #########################################################
#             print(f"\n2. Found {len(search_results)} search results")
#             # display_results(search_results)
#             for idx, result in enumerate(search_results, 1):
#                 print(f"\nResult {idx}:")
#                 print(f"Type: {result.type}")  # Added type display
#                 print(f"Content: {result.content}")
#                 print(f"Score: {result.score}")
#                 print(f"Strategy: {result.strategy}")
#             #########################################################

#             if not search_results:
#                 logger.info(f"No results found with {strategy.name} strategy")
#                 continue
                
#             # Select best result from current strategy results
#             candidate = self._select_candidate(search_results)
            
#             # Validate if the result is suitable for the query
#             is_suitable, reasoning = self._validate_result(candidate, user_query)
            
#             if is_suitable:
#                 logger.info(f"Found suitable result with {strategy.name} strategy: {candidate.metadata.get('name')}")
#                 logger.info(f"Reasoning: {reasoning}")
#                 best_result = candidate
#                 used_strategy = strategy
#                 break
                
#             logger.info(f"Result from {strategy.name} strategy was not suitable: {reasoning}")
        
#         # If we tried all strategies and none worked well, use the first strategy's top result
#         if best_result is None and attempted_strategies:
#             logger.warning("No suitable result found with any strategy, falling back to first strategy's top result")
#             first_strategy = strategies[0]
#             search_results = advanced_search.search(user_query, strategy=first_strategy)
#             if search_results:
#                 best_result = search_results[0]
#                 used_strategy = first_strategy
        
#         return best_result, used_strategy
    
#     def _select_candidate(self, search_results):
#         """Select the best candidate from the current search results"""
#         if len(search_results) == 1:
#             return search_results[0]
            
#         # Convert scores to standard Python float
#         for result in search_results:
#             if hasattr(result.score, 'item'):
#                 result.score = float(result.score)
                
#         # Take the highest scored result for now
#         # (We could also use Gemini to rank these, but the validation step will catch poor matches)
#         return max(search_results, key=lambda x: x.score)
        
#     def _validate_result(self, result, user_query):
#         """
#         Use Gemini to validate if the result is suitable for the query.
        
#         Returns:
#             (is_suitable, reasoning): Tuple of boolean and explanation
#         """
#         try:
#             # Extract relevant information
#             name = result.metadata.get('name', 'Unknown')
#             result_type = result.type
#             content = result.content[:2000]  # Limit content length
            
#             prompt = f"""
#             Evaluate if the following search result is suitable for answering the user's query.
            
#             User Query: "{user_query}"
            
#             Search Result:
#             - Name: {name}
#             - Type: {result_type}
#             - Content: {content}
            
#             Determine if this result directly addresses what the user is asking about.
#             Consider:
#             1. Is the entity type appropriate for the query?
#             2. Does the content contain information that would help answer the query?
#             3. Is there a good semantic match between the query and result?
            
#             Respond with a JSON object:
#             {{
#                 "is_suitable": true/false,
#                 "reasoning": "Your detailed explanation here"
#             }}
            
#             Your response should only include this JSON object, nothing else.
#             """
            
#             response = self.genai.generate_content(prompt)
#             response_text = response.text
            
#             # Extract JSON response
#             match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if match:
#                 result_json = json.loads(match.group(0))
#                 is_suitable = result_json.get("is_suitable", False)
#                 reasoning = result_json.get("reasoning", "No reasoning provided")
#                 return is_suitable, reasoning
                
#             return False, "Failed to parse Gemini response"
            
#         except Exception as e:
#             logger.error(f"Error validating result: {str(e)}")
#             return False, f"Validation error: {str(e)}"



# class AdaptiveSearchSelector:
#     def __init__(self, gemini_api_key):
#         self.gemini_api_key = gemini_api_key
#         self.genai = genai.GenerativeModel("gemini-1.5-pro")
#         genai.configure(api_key=self.gemini_api_key)
        
#     def select_best_result(self, advanced_search, user_query, initial_strategy=SearchStrategy.VECTOR):
#         """
#         Adaptively selects the best search result, trying different strategies if needed.
#         Validates all candidates in each strategy instead of just the highest scoring one.
        
#         Args:
#             advanced_search: AdvancedCodeSearch instance
#             user_query: Original user query
#             initial_strategy: Search strategy to try first
            
#         Returns:
#             best_result: The most relevant search result
#             used_strategy: The strategy that produced the best result
#         """
#         # List of strategies to try in order if initial results aren't good
#         strategies = [
#             initial_strategy,
#             SearchStrategy.HYBRID,
#             SearchStrategy.SEMANTIC,
#             SearchStrategy.ENSEMBLE
#         ]
        
#         # Make sure we don't try the same strategy twice
#         strategies = list(dict.fromkeys(strategies))
        
#         best_result = None
#         best_reasoning = None
#         best_score = -1.0
#         used_strategy = None
#         attempted_strategies = []
        
#         for strategy in strategies:
#             logger.info(f"Trying search strategy: {strategy.name}")
#             attempted_strategies.append(strategy.name)
            
#             # Perform search with current strategy
#             search_results = advanced_search.search(user_query, strategy=strategy)

#             #########################################################
#             print(f"\n2. Found {len(search_results)} search results")
#             # display_results(search_results)
#             for idx, result in enumerate(search_results, 1):
#                 print(f"\nResult {idx}:")
#                 print(f"Type: {result.type}")  # Added type display
#                 print(f"Content: {result.content}")
#                 print(f"Score: {result.score}")
#                 print(f"Strategy: {result.strategy}")
#             #########################################################

#             if not search_results:
#                 logger.info(f"No results found with {strategy.name} strategy")
#                 continue
                
#             # Standardize scores for all results
#             for result in search_results:
#                 if hasattr(result.score, 'item'):
#                     result.score = float(result.score)
            
#             # Validate all candidates in this strategy
#             strategy_best_result = None
#             strategy_best_reasoning = None
#             strategy_best_score = -1.0
            
#             for candidate in search_results:
#                 is_suitable, reasoning = self._validate_result(candidate, user_query)
                
#                 # Calculate a combined score based on search relevance and suitability
#                 # This can be adjusted based on your preferences
#                 combined_score = candidate.score * (1.0 if is_suitable else 0.5)
                
#                 logger.info(f"Candidate: {candidate.metadata.get('name')} - Score: {candidate.score} - Suitable: {is_suitable} - Combined: {combined_score}")
                
#                 # If this is the best result within this strategy so far, update
#                 if combined_score > strategy_best_score:
#                     strategy_best_result = candidate
#                     strategy_best_reasoning = reasoning
#                     strategy_best_score = combined_score
            
#             # If we found a suitable result in this strategy
#             if strategy_best_result and strategy_best_score > best_score:
#                 logger.info(f"Found better result with {strategy.name} strategy: {strategy_best_result.metadata.get('name')}")
#                 logger.info(f"Reasoning: {strategy_best_reasoning}")
#                 best_result = strategy_best_result
#                 best_reasoning = strategy_best_reasoning
#                 best_score = strategy_best_score
#                 used_strategy = strategy
                
#                 # If we found a suitable result with a high enough score, we can break early
#                 # You can adjust this threshold based on your requirements
#                 if best_score >= 0.8:  # Example threshold
#                     logger.info(f"Found highly suitable result with score {best_score}, breaking early")
#                     break
        
#         # If we tried all strategies and none worked well, use the best result we found across all strategies
#         if best_result is None and attempted_strategies:
#             logger.warning("No suitable result found with any strategy, falling back to first strategy's top result")
#             first_strategy = strategies[0]
#             search_results = advanced_search.search(user_query, strategy=first_strategy)
#             if search_results:
#                 best_result = search_results[0]
#                 used_strategy = first_strategy
        
#         return best_result, used_strategy
    
#     def _validate_result(self, result, user_query):
#         """
#         Use Gemini to validate if the result is suitable for the query.
        
#         Returns:
#             (is_suitable, reasoning): Tuple of boolean and explanation
#         """
#         try:
#             # Extract relevant information
#             name = result.metadata.get('name', 'Unknown')
#             result_type = result.type
#             content = result.content[:2000]  # Limit content length
            
#             prompt = f"""
#             Evaluate if the following search result is suitable for answering the user's query.
            
#             User Query: "{user_query}"
            
#             Search Result:
#             - Name: {name}
#             - Type: {result_type}
#             - Content: {content}
            
#             Determine if this result directly addresses what the user is asking about.
#             Consider:
#             1. Is the entity type appropriate for the query?
#             2. Does the content contain information that would help answer the query?
#             3. Is there a good semantic match between the query and result?
            
#             Respond with a JSON object:
#             {{
#                 "is_suitable": true/false,
#                 "reasoning": "Your detailed explanation here"
#             }}
            
#             Your response should only include this JSON object, nothing else.
#             """
            
#             response = self.genai.generate_content(prompt)
#             response_text = response.text
            
#             # Extract JSON response
#             match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if match:
#                 result_json = json.loads(match.group(0))
#                 is_suitable = result_json.get("is_suitable", False)
#                 reasoning = result_json.get("reasoning", "No reasoning provided")
#                 return is_suitable, reasoning
                
#             return False, "Failed to parse Gemini response"
            
#         except Exception as e:
#             logger.error(f"Error validating result: {str(e)}")
#             return False, f"Validation error: {str(e)}"



# import time
# from functools import lru_cache

# class AdaptiveSearchSelector:
#     def __init__(self, gemini_api_key):
#         self.gemini_api_key = gemini_api_key
#         self.genai = genai.GenerativeModel("gemini-1.5-pro")
#         genai.configure(api_key=self.gemini_api_key)
#         self.request_count = 0
#         self.last_request_time = 0
#         self.requests_per_minute = 60  # Adjust based on your API quota
        
#     def select_best_result(self, advanced_search, user_query, initial_strategy=SearchStrategy.VECTOR):
#         """
#         Adaptively selects the best search result with rate limiting to avoid quota exhaustion.
#         Only validates the top candidates from each strategy.
        
#         Args:
#             advanced_search: AdvancedCodeSearch instance
#             user_query: Original user query
#             initial_strategy: Search strategy to try first
            
#         Returns:
#             best_result: The most relevant search result
#             used_strategy: The strategy that produced the best result
#         """
#         # List of strategies to try in order
#         strategies = [
#             initial_strategy,
#             SearchStrategy.HYBRID,
#             SearchStrategy.SEMANTIC,
#             SearchStrategy.ENSEMBLE
#         ]
        
#         # Make sure we don't try the same strategy twice
#         strategies = list(dict.fromkeys(strategies))
        
#         best_result = None
#         best_reasoning = None
#         best_score = -1.0
#         used_strategy = None
#         attempted_strategies = []
        
#         for strategy in strategies:
#             logger.info(f"Trying search strategy: {strategy.name}")
#             attempted_strategies.append(strategy.name)
            
#             # Perform search with current strategy
#             search_results = advanced_search.search(user_query, strategy=strategy)

#             #########################################################
#             print(f"\n2. Found {len(search_results)} search results")
#             # display_results(search_results)
#             for idx, result in enumerate(search_results, 1):
#                 print(f"\nResult {idx}:")
#                 print(f"Type: {result.type}")  # Added type display
#                 print(f"Content: {result.content}")
#                 print(f"Score: {result.score}")
#                 print(f"Strategy: {result.strategy}")
#             #########################################################


#             if not search_results:
#                 logger.info(f"No results found with {strategy.name} strategy")
#                 continue
                
#             # Standardize scores
#             for result in search_results:
#                 if hasattr(result.score, 'item'):
#                     result.score = float(result.score)
            
#             # Sort results by score
#             search_results.sort(key=lambda x: x.score, reverse=True)
            
#             # Only validate the top N results from each strategy
#             max_to_validate = min(3, len(search_results))  # Adjust this number based on your needs
            
#             # Try validating the top results in order
#             for i in range(max_to_validate):
#                 candidate = search_results[i]
                
#                 # Skip validation for obviously poor matches (very low or negative scores)
#                 if candidate.score < 0.01:
#                     logger.info(f"Skipping validation for low-scoring candidate: {candidate.metadata.get('name')} - Score: {candidate.score}")
#                     continue
                
#                 # Use a cache key to avoid re-validating the same entity for similar queries
#                 cache_key = self._create_cache_key(candidate, user_query)
#                 is_suitable, reasoning = self._cached_validate_result(cache_key, candidate, user_query)
                
#                 # If we got rate limited, back off and try a simpler approach
#                 if reasoning and "rate limit" in reasoning.lower():
#                     logger.warning("Rate limit detected, falling back to score-based selection")
#                     # If rate limited, just use the highest scored result from this strategy
#                     return search_results[0], strategy
                
#                 combined_score = candidate.score * (1.0 if is_suitable else 0.5)
                
#                 logger.info(f"Candidate: {candidate.metadata.get('name')} - Score: {candidate.score} - Suitable: {is_suitable} - Combined: {combined_score}")
                
#                 if is_suitable and combined_score > best_score:
#                     logger.info(f"Found suitable result with {strategy.name} strategy: {candidate.metadata.get('name')}")
#                     logger.info(f"Reasoning: {reasoning}")
#                     best_result = candidate
#                     best_reasoning = reasoning
#                     best_score = combined_score
#                     used_strategy = strategy
                    
#                     # Break early if we found a good match
#                     if combined_score >= 0.8:  # Threshold for "good enough"
#                         return best_result, used_strategy
        
#         # If we found at least one suitable result across any strategy
#         if best_result:
#             return best_result, used_strategy
        
#         # If no suitable results were found but we have results, return the top scoring one
#         if attempted_strategies and any(advanced_search.search(user_query, strategy=s) for s in strategies):
#             logger.warning("No suitable result found, falling back to highest scoring result")
#             for strategy in strategies:
#                 results = advanced_search.search(user_query, strategy=strategy)
#                 if results:
#                     return max(results, key=lambda x: x.score), strategy
        
#         # If all else fails, return None
#         return None, None
    
#     def _rate_limit_check(self):
#         """
#         Implements a simple rate limiter to prevent API quota exhaustion.
#         """
#         current_time = time.time()
#         elapsed = current_time - self.last_request_time
        
#         # Reset counter if more than a minute has passed
#         if elapsed > 60:
#             self.request_count = 0
#             self.last_request_time = current_time
        
#         # If we're approaching the limit, sleep to avoid hitting it
#         if self.request_count >= self.requests_per_minute:
#             sleep_time = max(0, 60 - elapsed)
#             logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
#             time.sleep(sleep_time)
#             self.request_count = 0
#             self.last_request_time = time.time()
        
#         self.request_count += 1
#         self.last_request_time = current_time
    
#     def _create_cache_key(self, result, user_query):
#         """Create a cache key for the result and query"""
#         name = result.metadata.get('name', 'Unknown')
#         # Use a simpler version of the query to increase cache hits
#         simplified_query = ' '.join(word.lower() for word in user_query.split() if len(word) > 3)
#         return f"{name}:{simplified_query}"
    
#     @lru_cache(maxsize=100)
#     def _cached_validate_result(self, cache_key, result, user_query):
#         """
#         Cached version of the validation function to reduce API calls.
#         The cache_key is used for the lru_cache but isn't used in the function.
#         """
#         return self._validate_result(result, user_query)
    
#     def _validate_result(self, result, user_query):
#         """
#         Use Gemini to validate if the result is suitable for the query.
#         Implements rate limiting to avoid quota exhaustion.
        
#         Returns:
#             (is_suitable, reasoning): Tuple of boolean and explanation
#         """
#         try:
#             # Check if we need to rate limit
#             self._rate_limit_check()
            
#             # Extract relevant information
#             name = result.metadata.get('name', 'Unknown')
#             result_type = result.type
            
#             # Use a shorter content excerpt to reduce token usage
#             content = result.content[:1000] if hasattr(result, 'content') and result.content else "No content available"
            
#             prompt = f"""
#             Evaluate if this search result is suitable for answering this query.
            
#             Query: "{user_query}"
            
#             Result:
#             - Name: {name}
#             - Type: {result_type}
#             - Content excerpt: {content}
            
#             Is this result directly relevant to the query?
            
#             JSON response only:
#             {{
#                 "is_suitable": true/false,
#                 "reasoning": "brief explanation"
#             }}
#             """
            
#             response = self.genai.generate_content(prompt)
#             response_text = response.text
            
#             # Extract JSON response
#             match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if match:
#                 result_json = json.loads(match.group(0))
#                 is_suitable = result_json.get("is_suitable", False)
#                 reasoning = result_json.get("reasoning", "No reasoning provided")
#                 return is_suitable, reasoning
                
#             return False, "Failed to parse Gemini response"
            
#         except Exception as e:
#             error_msg = str(e)
#             logger.error(f"Error validating result: {error_msg}")
            
#             # For rate limit errors, return a specific message so we can detect it
#             if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
#                 return False, "Rate limit exceeded, try again later"
            
#             # For other errors, just proceed with score-based selection
#             return False, f"Validation error: {error_msg}"
            
#     def simple_score_based_selection(self, search_results):
#         """
#         Fallback method that selects the best result based only on score.
#         Used when API rate limits are encountered.
#         """
#         if not search_results:
#             return None
            
#         # Standardize scores
#         for result in search_results:
#             if hasattr(result.score, 'item'):
#                 result.score = float(result.score)
                
#         return max(search_results, key=lambda x: x.score)


import time
from functools import lru_cache

class AdaptiveSearchSelector:
    def __init__(self, gemini_api_key):
        self.gemini_api_key = gemini_api_key
        self.genai = genai.GenerativeModel("gemini-1.5-pro")
        genai.configure(api_key=self.gemini_api_key)
        self.request_count = 0
        self.last_request_time = 0
        self.requests_per_minute = 60  # Adjust based on your API quota
        self.validation_cache = {}  # Manual cache instead of lru_cache
        
    def select_best_result(self, advanced_search, user_query, initial_strategy=SearchStrategy.VECTOR):
        """
        Adaptively selects the best search result with rate limiting to avoid quota exhaustion.
        Only validates the top candidates from each strategy.
        
        Args:
            advanced_search: AdvancedCodeSearch instance
            user_query: Original user query
            initial_strategy: Search strategy to try first
            
        Returns:
            best_result: The most relevant search result
            used_strategy: The strategy that produced the best result
        """
        # List of strategies to try in order
        strategies = [
            initial_strategy,
            SearchStrategy.HYBRID,
            SearchStrategy.SEMANTIC,
            SearchStrategy.ENSEMBLE
        ]
        
        # Make sure we don't try the same strategy twice
        strategies = list(dict.fromkeys(strategies))
        
        best_result = None
        best_reasoning = None
        best_score = -1.0
        used_strategy = None
        attempted_strategies = []
        
        for strategy in strategies:
            logger.info(f"Trying search strategy: {strategy.name}")
            attempted_strategies.append(strategy.name)
            
            # Perform search with current strategy
            search_results = advanced_search.search(user_query, strategy=strategy)

            # Filter out results with type 'component'
            search_results = [result for result in search_results if result.type.lower() != 'component']

            #########################################################
            print(f"\n2. Found {len(search_results)} search results")
            # display_results(search_results)
            for idx, result in enumerate(search_results, 1):
                print(f"\nResult {idx}:")
                print(f"Type: {result.type}")  # Added type display
                print(f"Content: {result.content}")
                print(f"Score: {result.score}")
                print(f"Strategy: {result.strategy}")
            #########################################################

            if not search_results:
                logger.info(f"No results found with {strategy.name} strategy")
                continue
                
            # Standardize scores
            for result in search_results:
                if hasattr(result.score, 'item'):
                    result.score = float(result.score)
            
            # Sort results by score
            search_results.sort(key=lambda x: x.score, reverse=True)
            
            # Only validate the top N results from each strategy
            max_to_validate = min(3, len(search_results))  # Adjust this number based on your needs
            
            # Try validating the top results in order
            for i in range(max_to_validate):
                candidate = search_results[i]
                
                # Skip validation for obviously poor matches (very low or negative scores)
                if candidate.score < 0.01:
                    logger.info(f"Skipping validation for low-scoring candidate: {candidate.metadata.get('name')} - Score: {candidate.score}")
                    continue
                
                # Get entity name to use in cache key
                entity_name = candidate.metadata.get('name', 'Unknown')
                
                # Create cache key using entity name and query instead of the SearchResult object
                cache_key = self._create_cache_key(entity_name, user_query)
                
                # Check cache or validate
                is_suitable, reasoning = self._get_or_validate_result(cache_key, candidate, user_query)
                
                # If we got rate limited, back off and try a simpler approach
                if reasoning and "rate limit" in reasoning.lower():
                    logger.warning("Rate limit detected, falling back to score-based selection")
                    # If rate limited, just use the highest scored result from this strategy
                    return search_results[0], strategy
                
                combined_score = candidate.score * (1.0 if is_suitable else 0.5)
                
                logger.info(f"Candidate: {entity_name} - Score: {candidate.score} - Suitable: {is_suitable} - Combined: {combined_score}")
                
                if is_suitable and combined_score > best_score:
                    logger.info(f"Found suitable result with {strategy.name} strategy: {entity_name}")
                    logger.info(f"Reasoning: {reasoning}")
                    best_result = candidate
                    best_reasoning = reasoning
                    best_score = combined_score
                    used_strategy = strategy
                    
                    # Break early if we found a good match
                    if combined_score >= 0.8:  # Threshold for "good enough"
                        return best_result, used_strategy
        
        # If we found at least one suitable result across any strategy
        if best_result:
            return best_result, used_strategy
        
        # If no suitable results were found but we have results, return the top scoring one
        if attempted_strategies and any(advanced_search.search(user_query, strategy=s) for s in strategies):
            logger.warning("No suitable result found, falling back to highest scoring result")
            for strategy in strategies:
                results = advanced_search.search(user_query, strategy=strategy)
                if results:
                    return max(results, key=lambda x: x.score), strategy
        
        # If all else fails, return None
        return None, None
    
    def _rate_limit_check(self):
        """
        Implements a simple rate limiter to prevent API quota exhaustion.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Reset counter if more than a minute has passed
        if elapsed > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If we're approaching the limit, sleep to avoid hitting it
        if self.request_count >= self.requests_per_minute:
            sleep_time = max(0, 60 - elapsed)
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
        self.request_count += 1
        self.last_request_time = current_time
    
    def _create_cache_key(self, entity_name, user_query):
        """Create a cache key for the entity name and query"""
        # Use a simpler version of the query to increase cache hits
        simplified_query = ' '.join(word.lower() for word in user_query.split() if len(word) > 3)
        return f"{entity_name}:{simplified_query}"
    
    def _get_or_validate_result(self, cache_key, result, user_query):
        """
        Check if we have a cached validation result, otherwise validate and cache.
        """
        # Check if we have this result cached
        if cache_key in self.validation_cache:
            logger.info(f"Using cached validation result for {cache_key}")
            return self.validation_cache[cache_key]
        
        # If not cached, validate and cache the result
        validation_result = self._validate_result(result, user_query)
        
        # Store in cache (limit cache size to prevent memory issues)
        if len(self.validation_cache) > 100:
            # Remove a random key if cache gets too big
            self.validation_cache.pop(next(iter(self.validation_cache)))
        
        self.validation_cache[cache_key] = validation_result
        return validation_result
    
    def _validate_result(self, result, user_query):
        """
        Use Gemini to validate if the result is suitable for the query.
        Implements rate limiting to avoid quota exhaustion.
        
        Returns:
            (is_suitable, reasoning): Tuple of boolean and explanation
        """
        try:
            # Check if we need to rate limit
            self._rate_limit_check()
            
            # Extract relevant information
            name = result.metadata.get('name', 'Unknown')
            result_type = result.type
            
            # Use a shorter content excerpt to reduce token usage
            content = result.content[:1000] if hasattr(result, 'content') and result.content else "No content available"
            
            prompt = f"""
            Evaluate if this search result is suitable for answering this query.
            
            Query: "{user_query}"
            
            Result:
            - Name: {name}
            - Type: {result_type}
            - Content excerpt: {content}
            
            Is this result directly relevant to the query?
            
            JSON response only:
            {{
                "is_suitable": true/false,
                "reasoning": "brief explanation"
            }}
            """
            
            response = self.genai.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                result_json = json.loads(match.group(0))
                is_suitable = result_json.get("is_suitable", False)
                reasoning = result_json.get("reasoning", "No reasoning provided")
                return is_suitable, reasoning
                
            return False, "Failed to parse Gemini response"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error validating result: {error_msg}")
            
            # For rate limit errors, return a specific message so we can detect it
            if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                return False, "Rate limit exceeded, try again later"
            
            # For other errors, just proceed with score-based selection
            return False, f"Validation error: {error_msg}"
            
    def simple_score_based_selection(self, search_results):
        """
        Fallback method that selects the best result based only on score.
        Used when API rate limits are encountered.
        """
        if not search_results:
            return None
            
        # Standardize scores
        for result in search_results:
            if hasattr(result.score, 'item'):
                result.score = float(result.score)
                
        return max(search_results, key=lambda x: x.score)



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
    #########################################################################################
    #########################################################################################
    #########################################################################################
    def _create_analysis_prompt(self, user_query: str, path_selection: PathSelection, entity_contexts: List[str]) -> str:
        return f"""Analyze this code path to answer the user's query: "{user_query}"

    Detailed content of all entities in the path:

    {chr(10).join(entity_contexts)}

    Please provide a comprehensive analysis that:
    1. Directly addresses the user's query
    2. Explains how the components interact and depend on each other
    3. Describes the flow of data and control through the path
    4. Details the key functionality and purpose of each component
    5. Identifies any potential areas for attention or improvement
    6. Clearly connects your findings back to what the user wants to know

    Format your response as a detailed, well-structured analysis that thoroughly explains the code path and answers the user's query. Include specific examples and references to the code where relevant.

    Important aspects to cover:
    - Begin with a clear, direct answer to the user's query
    - Break down complex interactions between components
    - Highlight any relevant implementation details
    - Explain any important patterns or architectural decisions
    - Note any potential optimization opportunities or areas needing attention
    - Conclude with recommendations or best practices if applicable

    Keep the focus on helping the user understand exactly what they asked about while providing necessary context from the code path."""
    #########################################################################################
    #########################################################################################
    #########################################################################################
#     def _create_analysis_prompt(self, user_query: str, path_selection: PathSelection, entity_contexts: List[str]) -> str:
#         return f"""Analyze this code path based on the user's query: "{user_query}"

# Context about the selected path:
# Question asked: {path_selection.question_asked}
# User's response: {path_selection.user_response}
# Path confidence score: {path_selection.confidence_score}

# Detailed content of all entities in the path:

# {chr(10).join(entity_contexts)}

# Provide a comprehensive analysis focusing on:
# 1. How the components interact and depend on each other
# 2. The flow of data and control through the path
# 3. Key functionality and purpose of each component
# 4. Potential areas for attention or improvement
# 5. How this relates to the user's original query

# Give a detailed, prose-based analysis that explains the code path thoroughly."""
    #########################################################################################
    #########################################################################################
    #########################################################################################

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

    # def _get_path_selection(self, function_name: str, query: str, max_depth: int) -> Optional[PathSelection]:
    #     chains = self.retriever.get_complete_chains(
    #         node_name=function_name,
    #         max_depth=max_depth,
    #         node_type="CodeEntity"
    #     )
        
    #     if not chains:
    #         return None

    #     formatted_chains = self._format_chains(chains)
    #     grouped_chains = self._group_similar_paths(formatted_chains)
        
    #     return self.selector.select_path_interactively(
    #         list(grouped_chains.values())[0],
    #         function_name
    #     )

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




    ###############################################################
    ###############################################################
    ###############################################################
    @app.route('/clear_database', methods=['POST'])
    def clear_database():
        try:
            data = request.json
            db_type = data.get('type')

            if db_type == 'neo4j':
                # Create Neo4j session to clear the database
                neo4j_processor = Neo4jCodeEntityProcessor(
                    uri=os.environ.get('NEO4J_URI'),
                    username=os.environ.get('NEO4J_USERNAME'),
                    password=os.environ.get('NEO4J_PASSWORD')
                )
                
                with neo4j_processor.driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                
                neo4j_processor.close()
                message = "Neo4j database cleared successfully"

            elif db_type == 'vector':
                # Clear vector store indices
                if os.path.exists("vector_stores"):
                    shutil.rmtree("vector_stores")
                message = "Vector store cleared successfully"

            else:
                return jsonify({"error": "Invalid database type"}), 400

            return jsonify({"message": message})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/process_codebase', methods=['POST'])
    def process_codebase():
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400

            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Create temporary directory for uploaded files
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded file
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(temp_dir, filename)
            uploaded_file.save(file_path)

            # Process the uploaded codebase
            rdk_assistant = RDKAssistant(
                #code_base_path=temp_dir,
                code_base_path=os.getenv('CODE_BASE_PATH'),
                gemini_api_key=os.environ.get('GEMINI_API_KEY')
            )
            
            # Process the new code
            rdk_assistant._process_new_code(Path(file_path))

            # Clean up temporary files
            shutil.rmtree(temp_dir)

            return jsonify({
                "message": "Codebase processed successfully",
                "filename": filename
            })

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return jsonify({"error": str(e)}), 500
    ###############################################################
    ###############################################################
    ###############################################################

    # Modify the chat endpoint to handle the interactive flow
    @app.route('/chat', methods=['POST'])
    def chat():
        print("\n---------------------------- PRPL ASSIST INFO [/chat <- Endpoint, chat() <- Function, app_new2.py  <- File]--------------------------------")
        print("\n1. User Query Reached at Server Endpoint /chat !!!")
        query = request.json['query']
        print("User Query : ",query)
        try:
            print("Reached 1 \n")
            
            ############################################################
            ############################################################
            import ipdb;
            ipdb.set_trace()
            # search_results = config['advanced_search'].search(
            #     query, 
            #     strategy=config['search_strategy']
            # )
            # #==================================================== DEBUG
            # print(f"\n2. Found {len(search_results)} search results")
            # # display_results(search_results)
            # for idx, result in enumerate(search_results, 1):
            #     print(f"\nResult {idx}:")
            #     print(f"Type: {result.type}")  # Added type display
            #     print(f"Content: {result.content}")
            #     print(f"Score: {result.score}")
            #     print(f"Strategy: {result.strategy}")
            # #==================================================== DEBUG
            # if not search_results:
            #     return jsonify({
            #         'response': "No results found. Please try a different query.",
            #         'mermaid_diagram': None,
            #         'status': 'no_results'
            #     }), 200
            #################################################################
            ################################################################
                    # Instead of immediately performing a search, use the adaptive selector
            print("Starting adaptive search process\n")
            
            # The adaptive selector will try different strategies as needed
            best_result, used_strategy = config['adaptive_selector'].select_best_result(
                advanced_search=config['advanced_search'],
                user_query=query,
                initial_strategy=config['search_strategy']
            )
            
            if not best_result:
                return jsonify({
                    'response': "No relevant results found. Please try a different query.",
                    'mermaid_diagram': None,
                    'status': 'no_results'
                }), 200
            
            #==========================================================================
            # Use Gemini to select the best result instead of always taking the first one
            # best_result = config['result_selector'].select_best_result(search_results, query)
            first_result_name = best_result.metadata.get('name')
            first_result_type = best_result.type.lower() if best_result.type else None
            #==========================================================================

            # first_result_name = search_results[0].metadata.get('name')
            # first_result = search_results[0]
            # first_result_name = first_result.metadata.get('name')
            # first_result_type = first_result.type.lower() if first_result.type else None
            #==================================================== DEBUG
            # print(f"\n3. Selecting Top Entity from Vector Search results : {first_result_name}")
            # print(f"\n4. Analyzing function path for: {first_result_name}")
            print(f"\n3. Gemini selected best entity from search results: {first_result_name}")
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
        'path_selector': GeminiPathSelector(os.environ.get('GEMINI_API_KEY')),
        'adaptive_selector': AdaptiveSearchSelector(os.environ.get('GEMINI_API_KEY'))  # New adaptive selector
    }
    
    app = create_app(config)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))