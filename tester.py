import sys
import requests
import os
import google.auth
from google.oauth2.service_account import Credentials
import google.generativeai as genai

def test_basic_https():
    print("Testing basic HTTPS connection...")
    try:
        response = requests.get("https://www.google.com")
        print(f"✓ Basic HTTPS: Success (Status code: {response.status_code})")
    except Exception as e:
        print(f"✗ Basic HTTPS failed: {e}")

def test_google_auth():
    print("\nTesting Google authentication...")
    try:
        # Try to load credentials similar to your main app
        if os.path.exists('credentials.json'):
            credentials = Credentials.from_service_account_file('credentials.json')
            print("✓ Loaded credentials from file")
        else:
            print("✗ credentials.json not found")
            return False
        
        # Test if credentials are valid
        print("Testing if credentials are valid...")
        # This just accesses the credentials without making an API call
        project_id = credentials.project_id
        print(f"✓ Credentials appear valid (Project ID: {project_id})")
        return True
    except Exception as e:
        print(f"✗ Google auth failed: {e}")
        return False

def test_genai_connection():
    print("\nTesting Google Generative AI connection...")
    try:
        # Only run if we have credentials
        if not os.path.exists('credentials.json'):
            print("✗ Skipping: credentials.json not found")
            return
            
        credentials = Credentials.from_service_account_file('credentials.json')
        genai.configure(credentials=credentials)
        
        # Try to get model information (minimal API call)
        models = genai.list_models()
        print(f"✓ Successfully connected to GenAI API and retrieved models")
        
    except Exception as e:
        print(f"✗ GenAI connection failed: {e}")
        print("\nError details:", sys.exc_info())

def print_env_info():
    print("\nEnvironment Information:")
    print(f"Python version: {sys.version}")
    print(f"Requests version: {requests.__version__}")
    print(f"SSL module version: {sys.modules.get('ssl', None)}")
    # Print relevant environment variables
    for var in ['CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'GOOGLE_API_USE_CLIENT_CERTIFICATE']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

if __name__ == "__main__":
    print("SSL Certificate Verification Test\n" + "="*30)
    print_env_info()
    test_basic_https()
    auth_success = test_google_auth()
    if auth_success:
        test_genai_connection()
    
    print("\nTest complete. Check the results above for diagnosis.")