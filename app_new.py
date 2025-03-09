from neo4j_CodeContextRetriever import CodeContextRetriever
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from RDKAssistant_Class import RDKAssistant
import os

from UNDER_TEST.final.Neo4jPathRetriever import Neo4jPathRetriever
from VectorStoreManager import VectorStoreManager
from google.oauth2.service_account import Credentials
from google.generativeai import configure
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from UNDER_TEST.final.FIASS_Search import AdvancedCodeSearch
from UNDER_TEST.final.FIASS_Search import SearchStrategy
from UNDER_TEST.final.GeminiPathAnalyzer import analyze_function_path_with_gemini
from UNDER_TEST.final.main import display_results
from UNDER_TEST.final.GeminiPathSelector import GeminiPathSelector

app = Flask(__name__)
CORS(app)

#================================================== SAFE
# Use environment variables for configuration
# assistant = RDKAssistant(
#     code_base_path=os.environ.get('CODE_BASE_PATH', '/tmp/code_base'),
#     gemini_api_key=os.environ.get('GEMINI_API_KEY')
# )
# assistant.initialize()
# neo4j_context_retriever = CodeContextRetriever(
#     neo4j_uri=os.environ.get('NEO4J_URI'), 
#     neo4j_username=os.environ.get('NEO4J_USERNAME'), 
#     neo4j_password=os.environ.get('NEO4J_PASSWORD'), 
#     gemini_api_key=os.environ.get('GEMINI_API_KEY')
# )
#================================================== SAFE
#================================================== NEW
if os.path.exists('credentials.json'):
    credentials = Credentials.from_service_account_file('credentials.json')
else:
    # For production, use environment variables
    credentials_info = {
        "type": os.getenv("GOOGLE_TYPE"),
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
        "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
    }
    credentials = Credentials.from_service_account_info(credentials_info)
configure(credentials=credentials)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    credentials=credentials
)

vector_store = VectorStoreManager(embedding_model)

vector_store.load_indices(r"D:\pushing\vector_stores")

advanced_search = AdvancedCodeSearch(vector_store)

retriever = Neo4jPathRetriever(
    uri=os.environ.get('NEO4J_URI'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD')
)
#================================================== NEW
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    import ipdb; ipdb.set_trace()
    try:
        #=============================================== NEW
        search_strategy = SearchStrategy.VECTOR  # You can change this as needed
        print(f"Starting search with query: {query}")
        print(f"Vector store status: {vector_store.get_store_info()}")  # You'll need to implement this method
        print(f"\n{search_strategy.value.capitalize()} Strategy Search Results:")
        search_results = advanced_search.search(
            query, 
            strategy=search_strategy
        )
        print(f"Found {len(search_results)} search results")
        if not search_results:
            return jsonify({
                'response': "I couldn't find any information about the get_tunnel_params function in the codebase. "
                           "Please verify the function name or try rephrasing your question.",
                'mermaid_diagram': None,
                'status': 'no_results'
            }), 200  # Return 200 instead of 404 for better UX
        # Display search results
        display_results(search_results)

        # if 1:
        if search_results:
            first_result_name = search_results[0].metadata.get('name')

            #====================================== TEST
            # first_result_name='tunnel_operate'

            #====================================== TEST
            print(f"Analyzing function path for: {first_result_name}")
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
                function_name=first_result_name,
                # query="How does the get_tunnel_params function work?",
                query=query,
                max_depth=7
            )
        #=============================================== NEW
            return jsonify({
                'response': result["analysis"],
                'mermaid_diagram': result["mermaid_diagram"]
            })
        else:
            return jsonify({
                'error': 'No search results found'
            }), 404\
            
    except Exception as e:
        app.logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({
            'response': "I encountered an error while processing your request. Please try again.",
            'mermaid_diagram': None,
            'error': str(e),
            'status': 'error'
        }), 200  # Return 200 with error info instead of 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))