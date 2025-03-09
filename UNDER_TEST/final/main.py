# from GeminiPathAnalyzer import analyze_function_path_with_gemini
# from Neo4jPathRetriever import Neo4jPathRetriever
# from integrator import analyze_function_paths_interactive
# import os
# from typing import Optional, Dict
# from FIASS_Search import *

from UNDER_TEST.final.GeminiPathAnalyzer import analyze_function_path_with_gemini
from UNDER_TEST.final.Neo4jPathRetriever import Neo4jPathRetriever
from UNDER_TEST.final.integrator import analyze_function_paths_interactive
import os
from typing import Optional, Dict
from UNDER_TEST.final.FIASS_Search import *



# VECTOR = "vector"
# HYBRID = "hybrid"
# SEMANTIC = "semantic"
# ENSEMBLE = "ensemble"


def main():
    # Initialize advanced search
    advanced_search = AdvancedCodeSearch(vector_store)
    
    # User query
    #user_query = "List all the parameters of Dslite Object"
    user_query = "How the get_tunnel_params works?"

    retriever = Neo4jPathRetriever(
        uri=os.environ.get('NEO4J_URI'),
        username=os.environ.get('NEO4J_USERNAME'),
        password=os.environ.get('NEO4J_PASSWORD')
    )
    
    try:
        # Perform search with selected strategy
        search_strategy = SearchStrategy.VECTOR  # You can change this as needed
        print(f"\n{search_strategy.value.capitalize()} Strategy Search Results:")
        search_results = advanced_search.search(
            user_query, 
            strategy=search_strategy
        )
        
        # Display search results
        display_results(search_results)
        
        # If we have results, use the name of the first result for path analysis
        if search_results:
            first_result_name = search_results[0].metadata.get('name')
            #====================================== TEST
            first_result_name='tunnel_operate'
            #====================================== TEST
            # Run complete analysis with Gemini integration
            #====================================== SAFE
            # analysis = analyze_function_path_with_gemini(
            #     retriever=retriever,
            #     function_name=first_result_name,
            #     query=user_query,
            #     max_depth=7
            # )
            #====================================== SAFE
            
            #====================================== TEST
            result = analyze_function_path_with_gemini(
                retriever=retriever,
                function_name="get_tunnel_params",
                query="How does the get_tunnel_params function work?",
                max_depth=7
            )

            if result:
                print("\nAnalysis:", result["analysis"])
                print("\nMermaid Diagram:")
                print(result["mermaid_diagram"])
            #====================================== TEST

    except Exception as e:
        print(f"Error occurred during search and analysis: {str(e)}")

def display_results(results):
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result.content[:200]}...")  # Truncate long content
        print(f"Metadata: {result.metadata}")
        print(f"Score: {result.score:.4f}")
        print(f"Strategy: {result.strategy}")

if __name__ == "__main__":
    main()