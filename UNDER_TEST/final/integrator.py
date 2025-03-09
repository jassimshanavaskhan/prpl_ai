# import os
# from Neo4jPathRetriever import Neo4jPathRetriever
# from GeminiPathSelector import GeminiPathSelector
# from typing import List, Dict, Optional

import os
from UNDER_TEST.final.Neo4jPathRetriever import Neo4jPathRetriever
from UNDER_TEST.final.GeminiPathSelector import GeminiPathSelector
from typing import List, Dict, Optional


def analyze_function_paths_interactive(
    retriever: Neo4jPathRetriever, 
    function_name: str, 
    query: str, 
    max_depth: int = 7
) -> Optional[Dict]:
    """
    Analyze function paths using Neo4j with interactive path selection,
    without pre-filtering paths
    
    Args:
        retriever: Neo4jPathRetriever instance
        function_name: Name of the function to analyze
        query: Natural language context about what user is looking for
        max_depth: Maximum depth for path traversal
        
    Returns:
        Selected path with analysis information
    """
    # Get complete chains from Neo4j
    chains = retriever.get_complete_chains(
        node_name=function_name, 
        max_depth=max_depth, 
        node_type="CodeEntity"
        # node_type="Object"
    )
    
    if not chains:
        print(f"No paths found for function {function_name}")
        return None
    
    #==============================================================================
    # Print the complete chains before any filtering
    print("\nAll Complete Chains (Before Filtering):")
    retriever.print_complete_chains(chains, function_name)
    #==============================================================================
        
    # Format chains for the selector
    formatted_chains = [
        {
            'chain': chain,
            'relevance_score': 1.0,  # Default score since we're not pre-filtering
            'relevance_explanation': (
                "A potential path showing relationships and interactions "
                "with this component"
            )
        }
        for chain in chains
    ]
    
    # Group similar paths to avoid overwhelming the user
    def get_path_signature(chain):
        """Get a signature for the path based on node types and relationships"""
        return tuple(
            (step['from']['type'], step['relationship'], step['to']['type'])
            for step in chain['path_sequence']
        )
    
    # Group paths by their signatures
    path_groups = {}
    for chain_wrapper in formatted_chains:
        sig = get_path_signature(chain_wrapper['chain'])
        if sig not in path_groups:
            path_groups[sig] = []
        path_groups[sig].append(chain_wrapper)
    
    # Select representative paths from each group
    representative_chains = [
        group[0] for group in path_groups.values()
    ]


    #============================================================= ADDED TODAY
    # Print representative chains
    print("\nRepresentative Chains (One per unique path structure):")
    print(f"Found {len(representative_chains)} unique path structures")
    for i, chain_wrapper in enumerate(representative_chains, 1):
        print(f"\nRepresentative Chain {i}:")
        retriever.print_complete_chains([chain_wrapper['chain']], function_name)
        # Print the number of similar paths in this group
        sig = get_path_signature(chain_wrapper['chain'])
        similar_count = len(path_groups[sig])
        if similar_count > 1:
            print(f"  [This path structure appears {similar_count} times in the complete set]")
    #============================================================= ADDED TODAY
    
    # Interactively select the most relevant path
    selector = GeminiPathSelector(os.environ.get('GEMINI_API_KEY'))
    selection = selector.select_path_interactively(representative_chains, function_name)
    #============== RETRURN
    # return PathSelection(
    #     selected_path=paths[analysis['selected_path_index']],
    #     confidence_score=analysis['confidence_score'],
    #     reasoning=analysis['reasoning'],
    #     question_asked=question,
    #     user_response=user_response
    # )
    #======================
    # If the selected path was a representative of a group,
    # show all similar paths to the user
    selected_sig = get_path_signature(selection.selected_path['chain'])
    similar_paths = path_groups[selected_sig]
    
    if len(similar_paths) > 1:
        print("\nFound similar paths with the same structure:")
        retriever.print_complete_chains(
            [p['chain'] for p in similar_paths], 
            function_name
        )
    
    # Print results
    print(f"\nSelected Path Analysis for {function_name}:")
    print(f"Question Asked: {selection.question_asked}")
    print(f"Your Response: {selection.user_response}")
    print(f"Confidence Score: {selection.confidence_score}/10")
    print(f"Reasoning: {selection.reasoning}")
    print("\nSelected Path Details:")
    retriever.print_complete_chains([selection.selected_path['chain']], function_name)
    
    return selection
