from GeminiPathAnalyzer import analyze_function_path_with_gemini
from Neo4jPathRetriever import Neo4jPathRetriever
from integrator import analyze_function_paths_interactive
import os
from typing import Optional, Dict

def main():
    # Initialize the retriever

    retriever = Neo4jPathRetriever(
        uri=os.environ.get('NEO4J_URI'),
        username=os.environ.get('NEO4J_USERNAME'),
        password=os.environ.get('NEO4J_PASSWORD')
    )
    # import ipdb
    # ipdb.set_trace()

    # Analyze paths with interactive selection
    analysis = analyze_function_path_with_gemini(
        retriever=retriever,
        function_name="setup_dslite_interface",
        query="How the dslite starts",
        max_depth=7
    )

if __name__ == "__main__":
    main()