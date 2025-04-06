from enum import Enum, auto
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from langchain_community.vectorstores import FAISS
from logger import logger
import google.generativeai as genai
from collections import defaultdict
import re
from google.api_core.exceptions import ResourceExhausted
import time
import os
import groq

class SearchStrategy(Enum):
    VECTOR = auto()      # Pure vector similarity search
    SEMANTIC = auto()    # Semantic understanding with Gemini
    HYBRID = auto()      # Combination of vector and keyword search
    ENSEMBLE = auto()    # Multiple methods with voting

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    type: str
    
    def __post_init__(self):
        # Convert numpy types to Python native types for serialization
        if hasattr(self.score, 'item'):
            self.score = float(self.score)

class AdvancedCodeSearch:
    # def __init__(self, vector_store_manager, gemini_api_key=None):
    #     self.vector_store_manager = vector_store_manager
    #     self.gemini_api_key = gemini_api_key
    #     if gemini_api_key:
    #         genai.configure(api_key=gemini_api_key)
    #         self.model = genai.GenerativeModel("gemini-1.5-pro")
    

    # def generate_with_retry(self,prompt, max_retries=15, initial_delay=1):
    #     retries = 0
    #     while retries < max_retries:
    #         try:
    #             #return self.model.generate_content(prompt)
    #             return self.model.generate_content(prompt)
    #         except ResourceExhausted as e:
    #             retries += 1
    #             if retries >= max_retries:
    #                 raise
    #             # Exponential backoff
    #             delay = initial_delay * (2 ** (retries - 1))
    #             print(f"Rate limit hit, retrying in {delay} seconds...")
    #             time.sleep(delay)

    def __init__(self, vector_store_manager, gemini_api_key=None, groq_api_key=None):
        self.vector_store_manager = vector_store_manager
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        
        # Initialize Gemini if API key is provided
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Initialize Groq if API key is provided
        if groq_api_key:
            self.groq_client = groq.Client(api_key=groq_api_key)
            # Default to a powerful model, but you can change as needed
            self.groq_model = "llama3-70b-8192"
    
    def generate_with_retry(self, prompt, max_retries=15, initial_delay=1):
        retries = 0
        use_fallback = False
        
        while retries < max_retries:
            try:
                # Try with primary API (Gemini)
                if not use_fallback:
                    logger.info("Using Gemini API")
                    return self.model.generate_content(prompt)
                # Use fallback API (Groq)
                else:
                    logger.info("Using Groq API as fallback")
                    if not hasattr(self, 'groq_client'):
                        if not self.groq_api_key:
                            raise ValueError("Groq API key not provided for fallback")
                        self.groq_client = groq.Client(api_key=self.groq_api_key)
                    
                    # Call Groq API
                    chat_completion = self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    
                    # Format response to match Gemini's format or create a compatible wrapper
                    return self._format_groq_response(chat_completion)
                    
            except ResourceExhausted as e:
                retries += 1
                logger.warning(f"Rate limit hit with Gemini API, attempt {retries}/{max_retries}")
                
                # Switch to fallback on first rate limit
                if not use_fallback and self.groq_api_key:
                    logger.info("Switching to Groq API fallback")
                    use_fallback = True
                    # Reset retry counter when switching to fallback
                    retries = 0
                    continue
                
                if retries >= max_retries:
                    raise
                
                # Exponential backoff
                delay = initial_delay * (2 ** (retries - 1))
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    def _format_groq_response(self, groq_response):
        """Format Groq response to match Gemini's response format or create a compatible wrapper"""
        # This is a simplified example - you'll need to adapt based on how you use the response
        class GroqWrapper:
            def __init__(self, response_text):
                self.text = response_text
                # Add more attributes as needed to match Gemini's response structure
            
            def __str__(self):
                return self.text
        
        # Extract the content from Groq's response
        content = groq_response.choices[0].message.content
        return GroqWrapper(content)

    def search(self, query: str, strategy: SearchStrategy = SearchStrategy.VECTOR, 
               top_k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """
        Perform an advanced search using the specified strategy.
        
        Args:
            query: The search query
            strategy: The search strategy to use
            top_k: Maximum number of results to return
            threshold: Minimum relevance score threshold
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: '{query}' using {strategy.name} strategy")
        
        if strategy == SearchStrategy.VECTOR:
            return self._vector_search(query, top_k, threshold)
        elif strategy == SearchStrategy.SEMANTIC:
            return self._semantic_search(query, top_k, threshold)
        elif strategy == SearchStrategy.HYBRID:
            return self._hybrid_search(query, top_k, threshold)
        elif strategy == SearchStrategy.ENSEMBLE:
            return self._ensemble_search(query, top_k, threshold)
        else:
            logger.warning(f"Unknown search strategy: {strategy}. Falling back to vector search.")
            return self._vector_search(query, top_k, threshold)
            
    def _vector_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """Perform vector similarity search across all stores."""
        all_results = []
        
        # Search in each vector store
        logger.info(f"[PRPL_ASSIST_LOG] Searching in each vector store...")
        for store_type, store in self.vector_store_manager.vector_stores.items():
            if store is None:
                continue
                
            try:
                # Get raw results from FAISS
                raw_results = store.similarity_search_with_score(query, k=top_k)
                
                # Process and filter results
                for doc, score in raw_results:
                    # Convert similarity score (lower is better) to relevance score (higher is better)
                    # FAISS returns L2 distance, so we need to convert it to a similarity score
                    relevance_score = 1.0 / (1.0 + float(score))
                    
                    if relevance_score >= threshold:
                        all_results.append(SearchResult(
                            content=doc.page_content,
                            metadata=doc.metadata,
                            score=relevance_score,
                            type=store_type
                        ))
            except Exception as e:
                logger.error(f"Error searching {store_type} store: {str(e)}")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """Use Gemini to perform semantic understanding and scoring of results."""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not provided. Falling back to vector search.")
            return self._vector_search(query, top_k, threshold)
            
        # First get candidate results using vector search with relaxed threshold
        candidates = self._vector_search(query, top_k * 2, threshold * 0.7)
        
        if not candidates:
            return []
            
        # Prepare batch evaluation with Gemini
        results_with_semantic_scores = []
        
        # Process in smaller batches to avoid token limits
        batch_size = 3
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            try:
                # Create prompt for batch evaluation
                prompt = f"""Evaluate how well each of these code entities answers the following query:
                
                Query: "{query}"
                
                Entities:
                """
                
                for idx, candidate in enumerate(batch):
                    entity_name = candidate.metadata.get('name', 'Unknown')
                    entity_type = candidate.metadata.get('type', 'Unknown')
                    
                    # Trim content to avoid token limits
                    content = candidate.content[:800] if len(candidate.content) > 800 else candidate.content
                    
                    prompt += f"""
                    Entity {idx+1}:
                    - Name: {entity_name}
                    - Type: {entity_type}
                    - Content: {content}
                    """
                
                prompt += """
                For each entity, provide:
                1. A relevance score from 0.0 to 1.0 where 1.0 is perfectly relevant
                2. A brief explanation of why
                
                Format your response as a valid JSON array of objects:
                [
                  {
                    "entity_idx": 1,
                    "relevance_score": 0.85,
                    "explanation": "This entity directly addresses..."
                  },
                  ...
                ]
                """
                
                # Get response from Gemini
                #response = self.model.generate_content(prompt)
                response = self.generate_with_retry(prompt)
                response_text = response.text
                
                # Extract JSON from response
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    import json
                    evaluations = json.loads(json_match.group(0))
                    
                    # Update scores based on Gemini's evaluation
                    for eval_item in evaluations:
                        idx = eval_item.get('entity_idx', 0) - 1
                        if 0 <= idx < len(batch):
                            semantic_score = float(eval_item.get('relevance_score', 0.0))
                            # Combine vector score and semantic score
                            combined_score = 0.3 * batch[idx].score + 0.7 * semantic_score
                            
                            results_with_semantic_scores.append(SearchResult(
                                content=batch[idx].content,
                                metadata=batch[idx].metadata,
                                score=combined_score,
                                type=batch[idx].type
                            ))
                else:
                    # If JSON parsing fails, use original scores
                    results_with_semantic_scores.extend(batch)
                    
            except Exception as e:
                logger.error(f"Error in semantic scoring batch: {str(e)}")
                # Fall back to using original scores for this batch
                results_with_semantic_scores.extend(batch)
        
        # Filter and sort results
        filtered_results = [r for r in results_with_semantic_scores if r.score >= threshold]
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results[:top_k]
    
    def _hybrid_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """Combine vector search with keyword matching for better precision."""
        # Extract key terms from query
        keywords = self._extract_keywords(query)
        
        # Get vector search results
        vector_results = self._vector_search(query, top_k * 2, threshold * 0.7)
        
        # Score results with keyword matching
        hybrid_results = []
        for result in vector_results:
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_score(result.content, keywords)
            
            # Combine scores (70% vector, 30% keyword)
            combined_score = 0.7 * result.score + 0.3 * keyword_score
            
            if combined_score >= threshold:
                hybrid_results.append(SearchResult(
                    content=result.content,
                    metadata=result.metadata,
                    score=combined_score,
                    type=result.type
                ))
        
        # Sort by combined score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:top_k]
    
    def _ensemble_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """Run multiple search strategies and combine results with a voting system."""
        # Run different search strategies
        vector_results = self._vector_search(query, top_k, threshold)
        
        # Try semantic search if API key is available
        if self.gemini_api_key:
            semantic_results = self._semantic_search(query, top_k, threshold)
        else:
            semantic_results = []
            
        hybrid_results = self._hybrid_search(query, top_k, threshold)
        
        # Combine results using a weighted voting system
        result_map = {}  # Maps entity name to accumulated score
        result_objects = {}  # Maps entity name to the actual result object
        
        # Weight for each strategy
        weights = {
            'vector': 0.3,
            'semantic': 0.5,
            'hybrid': 0.2
        }
        
        # Process vector results
        for idx, result in enumerate(vector_results):
            name = result.metadata.get('name', f"unknown_{idx}")
            position_score = 1.0 - (idx / len(vector_results)) if len(vector_results) > 0 else 0
            result_map[name] = result_map.get(name, 0) + weights['vector'] * (result.score * 0.7 + position_score * 0.3)
            result_objects[name] = result
        
        # Process semantic results
        for idx, result in enumerate(semantic_results):
            name = result.metadata.get('name', f"unknown_{idx}")
            position_score = 1.0 - (idx / len(semantic_results)) if len(semantic_results) > 0 else 0
            result_map[name] = result_map.get(name, 0) + weights['semantic'] * (result.score * 0.7 + position_score * 0.3)
            result_objects[name] = result
            
        # Process hybrid results
        for idx, result in enumerate(hybrid_results):
            name = result.metadata.get('name', f"unknown_{idx}")
            position_score = 1.0 - (idx / len(hybrid_results)) if len(hybrid_results) > 0 else 0
            result_map[name] = result_map.get(name, 0) + weights['hybrid'] * (result.score * 0.7 + position_score * 0.3)
            result_objects[name] = result
            
        # Create ensemble results with normalized scores
        ensemble_results = []
        max_score = max(result_map.values()) if result_map else 1.0
        
        for name, score in result_map.items():
            # Normalize score
            normalized_score = score / max_score
            
            if normalized_score >= threshold:
                result = result_objects[name]
                ensemble_results.append(SearchResult(
                    content=result.content,
                    metadata=result.metadata,
                    score=normalized_score,
                    type=result.type
                ))
                
        # Sort by ensemble score
        ensemble_results.sort(key=lambda x: x.score, reverse=True)
        return ensemble_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key technical terms from the query."""
        # Simple keyword extraction using common technical terms or regex patterns
        common_terms = [
            # PRPL specific terms 
            r'\bprpl\b', r'\bbroadband\b', r'\bdevice\b', r'\bcodebas[e]?\b', r'\bfoundation\b',
            # Programming concepts
            r'\bfunction\b', r'\bmethod\b', r'\bclass\b', r'\bstruct\b', r'\bobject\b', 
            r'\bvariable\b', r'\bparameter\b', r'\breturn\b', r'\btype\b', r'\binterface\b',
            r'\bevent\b', r'\bhandler\b', r'\bcallback\b', r'\blistener\b', r'\btrigger\b',
            # Actions
            r'\bcreate\b', r'\bgenerate\b', r'\bupdate\b', r'\bdelete\b', r'\binsert\b',
            r'\bmodify\b', r'\bchange\b', r'\bremove\b', r'\badd\b', r'\bget\b', r'\bset\b',
            # Component types
            r'\bapi\b', r'\bcomponent\b', r'\bmodule\b', r'\bservice\b', r'\butility\b',
            r'\bhelper\b', r'\bmanager\b', r'\bcontroller\b', r'\bmodel\b', r'\bview\b',
            # Data structures
            r'\barray\b', r'\blist\b', r'\bdictionary\b', r'\bmap\b', r'\bset\b', r'\btree\b',
            r'\bgraph\b', r'\bqueue\b', r'\bstack\b', r'\bheap\b', r'\bhash\b', r'\btable\b',
            # Patterns and paradigms
            r'\bpattern\b', r'\bdesign\b', r'\barchitecture\b', r'\bframework\b', r'\blibrary\b',
            r'\bsdk\b', r'\bapi\b', r'\bcli\b', r'\bgui\b', r'\bui\b', r'\bux\b'
        ]
        
        keywords = []
        for term in common_terms:
            matches = re.findall(term, query.lower())
            keywords.extend(matches)
            
        # Also add any camelCase or snake_case terms as they're likely code identifiers
        code_identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+(?:_[a-z]+)+\b', query)
        keywords.extend(code_identifiers)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [k for k in keywords if not (k in seen or seen.add(k))]
        
        return unique_keywords
        
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate a score based on keyword presence and proximity."""
        if not keywords:
            return 0.5  # Neutral score if no keywords
            
        content_lower = content.lower()
        
        # Count keyword matches
        match_count = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        
        # Calculate basic match ratio
        match_ratio = match_count / len(keywords) if keywords else 0
        
        # Check for clustering/proximity of keywords
        proximity_bonus = 0
        if match_count >= 2:
            # Simple proximity check - find the closest pair of different keywords
            min_distance = float('inf')
            for i, kw1 in enumerate(keywords):
                if kw1.lower() not in content_lower:
                    continue
                    
                pos1 = content_lower.find(kw1.lower())
                
                for kw2 in keywords[i+1:]:
                    if kw2.lower() not in content_lower:
                        continue
                        
                    pos2 = content_lower.find(kw2.lower())
                    distance = abs(pos2 - pos1)
                    min_distance = min(min_distance, distance)
            
            # Convert distance to proximity bonus (closer = higher bonus)
            if min_distance < float('inf'):
                proximity_bonus = max(0, 0.2 - (min_distance / 1000))
        
        # Final score combines match ratio and proximity bonus
        return min(1.0, match_ratio * 0.8 + proximity_bonus)

    def focused_search(self, query: str, entity_type: str, top_k: int = 5) -> List[SearchResult]:
        """Search specifically within one entity type."""
        if entity_type not in self.vector_store_manager.vector_stores:
            logger.warning(f"Entity type '{entity_type}' not found in vector stores")
            return []
            
        store = self.vector_store_manager.vector_stores[entity_type]
        if store is None:
            logger.warning(f"Vector store for '{entity_type}' is None")
            return []
            
        try:
            raw_results = store.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in raw_results:
                # Convert to relevance score (higher is better)
                relevance_score = 1.0 / (1.0 + float(score))
                
                results.append(SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=relevance_score,
                    type=entity_type
                ))
                
            return results
        except Exception as e:
            logger.error(f"Error in focused search for {entity_type}: {str(e)}")
            return []
            
    def contextual_search(self, query: str, context: str, top_k: int = 5) -> List[SearchResult]:
        """Search with additional context to improve relevance."""
        # Enhance query with context
        enhanced_query = f"{query} {context}"
        
        # Use ensemble strategy for most robust results
        results = self._ensemble_search(enhanced_query, top_k)
        
        return results
    
    def search_by_component(self, query: str, component_name: str, top_k: int = 5) -> List[SearchResult]:
        """Search specifically within a component."""
        # First do a regular search
        results = self._ensemble_search(query, top_k * 3)
        
        # Filter for the specific component
        component_results = [
            r for r in results 
            if r.metadata.get('component', '').lower() == component_name.lower()
        ]
        
        # Sort by score
        component_results.sort(key=lambda x: x.score, reverse=True)
        
        return component_results[:top_k]
    
    def ai_guided_search(self, user_query: str, top_k: int = 5) -> List[SearchResult]:
        """Use Gemini to analyze the query and guide the search process."""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not provided. Falling back to ensemble search.")
            return self._ensemble_search(user_query, top_k)
        
        try:
            # Step 1: Analyze the query to understand intent and extract key entities
            analysis_prompt = f"""
            Analyze this query about PRPL (broadband device codebase by prpl foundation):
            
            "{user_query}"
            
            Provide:
            1. The likely intent (e.g., "find a function", "understand component structure", "fix a bug")
            2. Key technical entities mentioned (functions, classes, components, etc.)
            3. The most relevant entity types to search (function, struct, component, api, odl_object, documentation)
            4. A refined search query that would find the most relevant code
            
            Format as JSON:
            {{
                "intent": "...",
                "key_entities": ["...", "..."],
                "entity_types_to_search": ["...", "..."],
                "refined_query": "..."
            }}
            """
            
            #response = self.model.generate_content(analysis_prompt)
            response = self.generate_with_retry(analysis_prompt)
            response_text = response.text
            
            # Extract JSON
            import json
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning("Failed to parse Gemini response. Falling back to ensemble search.")
                return self._ensemble_search(user_query, top_k)
                
            analysis = json.loads(json_match.group(0))
            
            # Step 2: Perform targeted searches based on the analysis
            all_results = []
            
            # Search in recommended entity types
            entity_types = analysis.get('entity_types_to_search', ['function', 'component'])
            refined_query = analysis.get('refined_query', user_query)
            
            # Perform focused searches for each recommended entity type
            for entity_type in entity_types:
                if entity_type in self.vector_store_manager.vector_stores:
                    type_results = self.focused_search(refined_query, entity_type, top_k)
                    all_results.extend(type_results)
            
            # Step 3: Perform additional search for key entities if mentioned
            key_entities = analysis.get('key_entities', [])
            for entity in key_entities:
                entity_query = f"{entity} {refined_query}"
                entity_results = self._ensemble_search(entity_query, max(2, top_k // 2))
                all_results.extend(entity_results)
            
            # Step 4: Deduplicate and rank final results
            deduplicated = {}
            for result in all_results:
                name = result.metadata.get('name', '')
                if name not in deduplicated or result.score > deduplicated[name].score:
                    deduplicated[name] = result
            
            final_results = list(deduplicated.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            # Step 5: Perform semantic reranking on top results
            if len(final_results) > top_k:
                # Take top results and rerank
                candidates = final_results[:min(top_k * 2, len(final_results))]
                
                reranking_prompt = f"""
                Rank these code entities based on how relevant they are to this query:
                
                Query: "{user_query}"
                
                Entities:
                """
                
                for idx, result in enumerate(candidates):
                    name = result.metadata.get('name', 'Unknown')
                    entity_type = result.metadata.get('type', 'Unknown')
                    content_excerpt = result.content[:500] + "..." if len(result.content) > 500 else result.content
                    
                    reranking_prompt += f"""
                    Entity {idx+1}:
                    - Name: {name}
                    - Type: {entity_type}
                    - Content: {content_excerpt}
                    """
                
                reranking_prompt += """
                Return a JSON array of ranked entity indices, with most relevant first:
                [2, 5, 1, ...]
                """
                
                #reranking_response = self.model.generate_content(reranking_prompt)
                reranking_response = self.generate_with_retry(reranking_prompt)
                reranking_text = reranking_response.text
                
                # Extract JSON array
                rank_match = re.search(r'\[([\d\s,]+)\]', reranking_text)
                if rank_match:
                    try:
                        # Parse and adjust indices (subtract 1)
                        ranks = [int(idx.strip()) - 1 for idx in rank_match.group(1).split(',') if idx.strip()]
                        
                        # Reorder based on ranks, filtering valid indices
                        valid_ranks = [r for r in ranks if 0 <= r < len(candidates)]
                        reranked = [candidates[r] for r in valid_ranks]
                        
                        # Add any remaining candidates not in the ranking
                        ranked_indices = set(valid_ranks)
                        remaining = [candidates[i] for i in range(len(candidates)) if i not in ranked_indices]
                        
                        final_results = reranked + remaining
                    except Exception as e:
                        logger.error(f"Error parsing reranking response: {str(e)}")
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in AI-guided search: {str(e)}")
            # Fall back to ensemble search
            return self._ensemble_search(user_query, top_k)