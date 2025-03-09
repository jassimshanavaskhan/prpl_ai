from neo4j_CodeContextRetriever import CodeContextRetriever
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from RDKAssistant_Class import RDKAssistant
import os

app = Flask(__name__)
CORS(app)

# Use environment variables for configuration
assistant = RDKAssistant(
    code_base_path=os.environ.get('CODE_BASE_PATH', '/tmp/code_base'),
    gemini_api_key=os.environ.get('GEMINI_API_KEY')
)
assistant.initialize()
neo4j_context_retriever = CodeContextRetriever(
    neo4j_uri=os.environ.get('NEO4J_URI'), 
    neo4j_username=os.environ.get('NEO4J_USERNAME'), 
    neo4j_password=os.environ.get('NEO4J_PASSWORD'), 
    gemini_api_key=os.environ.get('GEMINI_API_KEY')
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    try:
        result = neo4j_context_retriever.process_query(query)
        return jsonify({
            'response': result["response"],
            'mermaid_diagram': result.get("mermaid_diagram")  # Include mermaid diagram
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))