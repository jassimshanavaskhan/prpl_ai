import os
import subprocess
import tempfile
import google.generativeai as genai
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CodeContextRetriever:
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str, 
                 gemini_api_key: str):
        """
        Initialize Neo4j and Gemini connections
        
        :param neo4j_uri: Neo4j database URI
        :param neo4j_username: Neo4j username
        :param neo4j_password: Neo4j password
        :param gemini_api_key: Google Gemini API key
        """
        # Neo4j Connection
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
            print("Neo4j Driver Initialized Successfully!")
        except Exception as e:
            print(f"Error initializing Neo4j driver: {e}")
            raise

        # Gemini Connection
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')

    def generate_mermaid_sequence_diagram(self, context: Dict[str, Any]) -> str:
        """
        Generate an enhanced Mermaid sequence diagram with AI-powered context understanding
        
        :param context: Function context dictionary
        :return: Detailed Mermaid sequence diagram string
        """
        try:
            # Prepare context for Gemini analysis
            context_prompt = f"""
            Analyze the following code context and help me create a detailed Mermaid sequence diagram:

            Function Details:
            {context.get('function_details', 'No function details')}

            Incoming Calls:
            {context.get('incoming_calls', 'No incoming calls')}

            Outgoing Calls:
            {context.get('outgoing_calls', 'No outgoing calls')}

            For the sequence diagram, please:
            1. Identify the key actors/participants
            2. Determine the sequence and flow of interactions
            3. Add meaningful annotations or notes
            4. Highlight any complex interactions or decision points
            5. Use descriptive labels for calls and interactions
            6. Consider the system architecture and component relationships

            Provide a detailed description of how the sequence diagram should be structured.
            """

            # Generate diagram description using Gemini
            diagram_description_response = self.gemini_model.generate_content(context_prompt)
            diagram_description = diagram_description_response.text

            # Extract Mermaid sequence diagram using another Gemini prompt
            mermaid_prompt = f"""
            Based on this context and description, create a Mermaid sequence diagram:

            Context Description:
            {diagram_description}

            Guidelines for Mermaid Diagram:
            - Use clear, descriptive participant names
            - Show critical interactions and data flow
            - Use different types of arrows to represent interaction types
            - Add notes for complex interactions
            - Represent decision points and conditional flows
            - Ensure the diagram is readable and informative

            Provide ONLY the Mermaid sequence diagram code, without any additional text.
            """

            mermaid_response = self.gemini_model.generate_content(mermaid_prompt)
            mermaid_diagram = mermaid_response.text.strip()

            # Validate and enhance the generated diagram
            if not mermaid_diagram.startswith('sequenceDiagram'):
                # Fallback to a basic diagram if generation fails
                mermaid_diagram = self.generate_fallback_diagram(context)

            return mermaid_diagram

        except Exception as e:
            print(f"Error generating advanced sequence diagram: {e}")
            return self.generate_fallback_diagram(context)
        
    #========================================================================================================== SAFEEST JASS 
    def generate_fallback_diagram(self, context: Dict[str, Any]) -> str:
        """
        Generate a fallback sequence diagram when AI generation fails
        
        :param context: Function context dictionary
        :return: Basic Mermaid sequence diagram
        """
        # Start the sequence diagram
        mermaid_diagram = ["sequenceDiagram"]
        
        # Main function as central participant
        main_function = context.get('function_details', {}).get('name', 'MainFunction')
        mermaid_diagram.append(f"    participant {main_function}")
        
        # Add incoming calls with basic interactions
        if context.get('incoming_calls'):
            for call in context['incoming_calls']:
                caller = call.get('caller_name', 'UnknownCaller')
                mermaid_diagram.append(f"    participant {caller}")
                mermaid_diagram.append(f"    {caller}->>{main_function}: Invoke")
        
        # Add outgoing calls
        if context.get('outgoing_calls'):
            for call in context['outgoing_calls']:
                callee = call.get('callee_name', 'UnknownCallee')
                mermaid_diagram.append(f"    participant {callee}")
                mermaid_diagram.append(f"    {main_function}-->>{callee}: Call")
                
                # Add basic note if parameters are available
                if call.get('parameters'):
                    mermaid_diagram.append(f"    note right of {main_function}: Parameters: {call['parameters']}")
        
        return "\n".join(mermaid_diagram)
    #========================================================================================================== SAFEEST JASS 
    # def generate_fallback_diagram(self, context: Dict[str, Any]) -> str:
    #     """
    #     Generate a fallback sequence diagram when AI generation fails
        
    #     :param context: Function context dictionary
    #     :return: Basic Mermaid sequence diagram
    #     """
    #     # Start the sequence diagram
    #     mermaid_diagram = ["sequenceDiagram"]
        
    #     # Main function as central participant
    #     main_function = context.get('function_details', {}).get('name', 'MainFunction')
    #     mermaid_diagram.append(f"    participant {main_function}")
        
    #     # Add incoming calls with basic interactions
    #     if context.get('incoming_calls'):
    #         for call in context['incoming_calls']:
    #             caller = call.get('caller_name', 'UnknownCaller')
    #             mermaid_diagram.append(f"    participant {caller}")
    #             mermaid_diagram.append(f"    {caller}->>{main_function}: Invoke")
                
    #             # Add parameters note if available
    #             if call.get('parameters'):
    #                 param_str = str(call['parameters']).replace('"', "'")
    #                 mermaid_diagram.append(f"    note right of {caller}: Parameters: {param_str}")
        
    #     # Add outgoing calls
    #     if context.get('outgoing_calls'):
    #         for call in context['outgoing_calls']:
    #             callee = call.get('callee_name', 'UnknownCallee')
    #             mermaid_diagram.append(f"    participant {callee}")
    #             mermaid_diagram.append(f"    {main_function}-->>{callee}: Call")
                
    #             # Add parameters note if available
    #             if call.get('parameters'):
    #                 param_str = str(call['parameters']).replace('"', "'")
    #                 mermaid_diagram.append(f"    note right of {main_function}: Parameters: {param_str}")
        
    #     return "\n".join(mermaid_diagram)
    #========================================================================================================== TESTING JASS 


    # def save_sequence_diagram(self, mermaid_diagram: str) -> Optional[str]:
    #     """
    #     Save and render Mermaid sequence diagram
        
    #     :param mermaid_diagram: Mermaid diagram string
    #     :return: Path to generated diagram image or None
    #     """
    #     try:
    #         # Create a temporary file for the Mermaid diagram
    #         with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp_mmd:
    #             temp_mmd.write(mermaid_diagram)
    #             temp_mmd_path = temp_mmd.name
            
    #         # Create a temporary output file for the diagram
    #         with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as temp_png:
    #             temp_png_path = temp_png.name
            
    #         # Use Mermaid CLI to generate the diagram
    #         try:
    #             subprocess.run([
    #                 'mmdc', 
    #                 '-i', temp_mmd_path, 
    #                 '-o', temp_png_path, 
    #                 '-t', 'dark'  # Optional: choose a theme
    #             ], check=True)
                
    #             return temp_png_path
    #         except subprocess.CalledProcessError as e:
    #             print(f"Error generating diagram: {e}")
    #             return None
    #         except FileNotFoundError:
    #             print("Mermaid CLI (mmdc) not found. Please install it using 'npm install -g @mermaid-js/mermaid-cli'")
    #             return None
        
    #     except Exception as e:
    #         print(f"Error in diagram generation: {e}")
    #         return None
    def save_sequence_diagram(self, mermaid_diagram: str) -> Optional[str]:
        try:
            # Use full path to mmdc (adjust based on your Node.js installation)
            mmdc_path = r'C:\Users\39629\AppData\Roaming\npm\mmdc.cmd'  # Typical Windows path
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp_mmd:
                temp_mmd.write(mermaid_diagram)
                temp_mmd_path = temp_mmd.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as temp_png:
                temp_png_path = temp_png.name
            
            # Run mmdc with full path
            try:
                subprocess.run([
                    mmdc_path, 
                    '-i', temp_mmd_path, 
                    '-o', temp_png_path, 
                    '-t', 'dark'
                ], check=True, shell=True)
                
                return temp_png_path
            except subprocess.CalledProcessError as e:
                print(f"Error generating diagram: {e}")
                return None
        except Exception as e:
            print(f"Error in diagram generation: {e}")
            return None


    def search_functions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for functions based on user query using TF-IDF and cosine similarity
        
        :param query: User's search query
        :param top_k: Number of top matching functions to return
        :return: List of top matching functions
        """
        with self.neo4j_driver.session() as session:
            # Retrieve all functions
            result = session.run("""
                MATCH (e:CodeEntity {type: 'function'})
                RETURN e.name AS name, e.content AS content, e.file_path AS file_path
            """)
            
            # Convert to list of dicts
            functions = [
                {
                    'name': record['name'], 
                    'content': record['content'],
                    'file_path': record['file_path']
                } 
                for record in result
            ]

        # Prepare documents for TF-IDF
        documents = [func['content'] for func in functions]
        documents.append(query)  # Add query to vectorizer

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        # Sort functions by similarity
        similar_func_indices = cosine_similarities.argsort()[::-1][:top_k]
        
        return [functions[idx] for idx in similar_func_indices]

    def get_function_context(self, function_name: str) -> Dict[str, Any]:
        """
        Retrieve detailed context for a specific function
        
        :param function_name: Name of the function
        :return: Dictionary with function context
        """
        with self.neo4j_driver.session() as session:
            # Get function details
            function_query = """
            MATCH (e:CodeEntity {name: $function_name})
            RETURN e
            """
            function_result = session.run(function_query, {"function_name": function_name}).single()
            
            if not function_result:
                return {}

            # Get incoming calls
            incoming_calls_query = """
            MATCH (caller:CodeEntity)-[r:CALLS]->(func:CodeEntity {name: $function_name})
            RETURN caller.name AS caller_name, 
                   r.line_number AS line_number, 
                   r.parameters AS parameters
            """
            incoming_calls = list(session.run(incoming_calls_query, {"function_name": function_name}))

            # Get outgoing calls
            outgoing_calls_query = """
            MATCH (func:CodeEntity {name: $function_name})-[r:CALLS]->(callee:CodeEntity)
            RETURN callee.name AS callee_name, 
                   r.line_number AS line_number, 
                   r.parameters AS parameters
            """
            outgoing_calls = list(session.run(outgoing_calls_query, {"function_name": function_name}))

            return {
                'function_details': dict(function_result['e']),
                'incoming_calls': [dict(call) for call in incoming_calls],
                'outgoing_calls': [dict(call) for call in outgoing_calls]
            }

    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """
        Generate a context-aware response using Gemini
        
        :param query: User's original query
        :param context: Context retrieved from code analysis
        :return: Generated response
        """
        # Prepare context for Gemini prompt
        context_str = self.format_context(context)
        
        # Construct comprehensive prompt
        prompt = f"""
        User Query: {query}

        Code Context:
        {context_str}

        Please provide a detailed and informative response that:
        1. Directly addresses the user's query
        2. Uses the code context to provide insights
        3. Explains technical details clearly
        4. Highlights relevant code relationships
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context into a readable string
        
        :param context: Context dictionary
        :return: Formatted context string
        """
        formatted = []

        # Function Details
        if context.get('function_details'):
            formatted.append("Function Details:")
            for key, value in context['function_details'].items():
                formatted.append(f"- {key}: {value}")
        
        # Incoming Calls
        if context.get('incoming_calls'):
            formatted.append("\nIncoming Calls:")
            for call in context['incoming_calls']:
                formatted.append(f"- Caller: {call.get('caller_name', 'Unknown')}")
                formatted.append(f"  Line Number: {call.get('line_number', 'N/A')}")
                formatted.append(f"  Parameters: {call.get('parameters', 'N/A')}")
        
        # Outgoing Calls
        if context.get('outgoing_calls'):
            formatted.append("\nOutgoing Calls:")
            for call in context['outgoing_calls']:
                formatted.append(f"- Callee: {call.get('callee_name', 'Unknown')}")
                formatted.append(f"  Line Number: {call.get('line_number', 'N/A')}")
                formatted.append(f"  Parameters: {call.get('parameters', 'N/A')}")
        
        return "\n".join(formatted)


    #-----------------------------------------------------------------------------------------
    # def process_query(self, query: str) -> str:
    #     """
    #     Main method to process user query and generate response
        
    #     :param query: User's query
    #     :return: Generated response
    #     """
    #     # Search for most relevant functions
    #     similar_functions = self.search_functions(query)
        
    #     # If no similar functions found, return a default message
    #     if not similar_functions:
    #         return "No relevant functions found for the given query."
        
    #     # Get context for the top matching function
    #     top_function = similar_functions[0]
    #     function_context = self.get_function_context(top_function['name'])
        
    #     # Generate response
    #     response = self.generate_response(query, function_context)
        
    #     return response
    # Modify the process_query method in CodeContextRetriever
    #-----------------------------------------------------------------------------------------
    # def process_query(self, query: str) -> str:
    #     """
    #     Main method to process user query and generate response with sequence diagram
        
    #     :param query: User's query
    #     :return: Generated response with optional sequence diagram
    #     """
    #     # Search for most relevant functions
    #     similar_functions = self.search_functions(query)
        
    #     # If no similar functions found, return a default message
    #     if not similar_functions:
    #         return "No relevant functions found for the given query."
        
    #     # Get context for the top matching function
    #     top_function = similar_functions[0]
    #     function_context = self.get_function_context(top_function['name'])
        
    #     # Generate response
    #     response = self.generate_response(query, function_context)
        
    #     # Generate sequence diagram
    #     diagram_generator = SequenceDiagramGenerator()
    #     diagram_path = diagram_generator.generate_sequence_diagram(function_context)
        
    #     # Enhance response with diagram reference
    #     enhanced_response = diagram_generator.enhance_response_with_diagram(response, diagram_path)
        
    #     return enhanced_response
    #--------------------------------------------------------------------------------------------
    # Modify the CodeContextRetriever's process_query method
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced process_query to include sequence diagram generation
        
        :param query: User's query
        :return: Dictionary with response, context, and diagram path
        """
        # Search for most relevant functions
        similar_functions = self.search_functions(query)
        
        # If no similar functions found, return a default message
        if not similar_functions:
            return {
                "response": "No relevant functions found for the given query.",
                "context": None,
                "diagram_path": None
            }
        
        # Get context for the top matching function
        top_function = similar_functions[0]
        function_context = self.get_function_context(top_function['name'])
        
        # Generate response
        response = self.generate_response(query, function_context)
        
        # Generate sequence diagram
        mermaid_diagram = self.generate_mermaid_sequence_diagram(function_context)
        diagram_path = self.save_sequence_diagram(mermaid_diagram)
        
        # return {
        #     "response": response,
        #     "context": function_context,
        #     "diagram_path": diagram_path
        # }
        return {
            "response": response,
            "context": function_context,
            "diagram_path": diagram_path,
            "mermaid_diagram": mermaid_diagram  
        }

    def close(self):
        """
        Close database connections
        """
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
            print("Neo4j Driver Connection Closed.")

# def main():
#     # Configuration - Replace with your actual credentials
#     NEO4J_URI = "bolt://localhost:7687"
#     NEO4J_USERNAME = "neo4j"
#     NEO4J_PASSWORD = "Jassim@123"
#     GEMINI_API_KEY = "AIzaSyAGrz7Fw9flS5OnHu5G-EqvwT1pPVkWV64"

#     try:
#         # Initialize the context retriever
#         retriever = CodeContextRetriever(
#             NEO4J_URI, 
#             NEO4J_USERNAME, 
#             NEO4J_PASSWORD, 
#             GEMINI_API_KEY
#         )

#         # Interactive query loop
#         while True:
#             query = input("Enter your query (or 'quit' to exit): ")
            
#             if query.lower() == 'quit':
#                 break
            
#             # Process query
#             result = retriever.process_query(query)
            
#             # Print response
#             print("\nResponse:\n", result["response"])
            
#             # Display diagram path if generated
#             if result["diagram_path"]:
#                 print(f"\nSequence Diagram generated: {result['diagram_path']}")
            
#             print("\n" + "="*50 + "\n")

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         retriever.close()

# if __name__ == "__main__":
#     main()