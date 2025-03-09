from RDKAssistant_Class import RDKAssistant
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    assistant = RDKAssistant(
        code_base_path=os.getenv('CODE_BASE_PATH'),
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        neo4j_uri='bolt://localhost:7687',
        neo4j_username='neo4j',
        neo4j_password='Jassim@123'
    )
    assistant.initialize()
    assistant.handle_user_interaction()
    assistant.generate_code_flow_visualization('rdk_code_flow.png')
    assistant.close()