from RDKAssistant_Class import RDKAssistant
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    assistant = RDKAssistant(
        code_base_path=os.getenv('CODE_BASE_PATH'),
        gemini_api_key=os.getenv('GEMINI_API_KEY')
    )
    #import ipdb; ipdb.set_trace()
    assistant.initialize()
    assistant.handle_user_interaction()