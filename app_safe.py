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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    try:
        response = assistant.handle_user_query(query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))