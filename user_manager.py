from flask import Flask, request, jsonify
import subprocess
import os
import uuid
import random
import string
import json
import time
from pathlib import Path
import shutil

app = Flask(__name__)

# Configuration
INSTANCES_DIR = 'user_instances'
BASE_PORT = 9000
USER_DB_FILE = 'users.json'
API_KEYS = {}  # Load from environment or config file

# Load API keys from environment variables
API_KEYS['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY')
API_KEYS['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')

# Create instances directory if it doesn't exist
os.makedirs(INSTANCES_DIR, exist_ok=True)

# Load existing users from file
def load_users():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save users to file
def save_users(users):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f)

# Generate a random password
def generate_password():
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))

# Fix for the docker-compose file path issue in create_instance function
@app.route('/create_instance', methods=['POST'])
def create_instance():
    """Create a new user instance"""
    try:
        users = load_users()
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())[:8]
        
        # Find available ports
        neo4j_http_port = find_available_port(BASE_PORT)
        neo4j_bolt_port = find_available_port(neo4j_http_port + 1)
        app_port = find_available_port(neo4j_bolt_port + 1)
        
        # Generate Neo4j password
        neo4j_password = generate_password()
        
        # Create user directory (absolute path inside container)
        user_dir = os.path.join(os.path.abspath(INSTANCES_DIR), user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        print(f"Created user directory: {user_dir}")  # Debug log
        
        # Copy application files
        app_files = [
            'app2.py', 'Dockerfile.app', 'requirements.txt',
            'UNDER_TEST', 'templates', 'RDKAssistant_Class.py',
            'neo4j_Class.py', 'VectorStoreManager.py', 'ContentGenerator.py',
            'logger.py'
        ]
        
        for file in app_files:
            source = file
            destination = os.path.join(user_dir, file)
            
            print(f"Copying {source} to {destination}")  # Debug log
            
            try:
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                    print(f"Copied directory: {source}")
                elif os.path.isfile(source):
                    shutil.copy2(source, destination)
                    print(f"Copied file: {source}")
                else:
                    print(f"Warning: {source} not found")
            except Exception as e:
                print(f"Error copying {source}: {str(e)}")
        
        # Create code directory
        os.makedirs(os.path.join(user_dir, 'code'), exist_ok=True)
        print("Created code directory")  # Debug log
        
        # Copy Docker Compose template
        with open('docker-compose-template.yml', 'r') as f:
            template = f.read()
        
        # Create Docker Compose file with user-specific values
        docker_compose_file = os.path.join(user_dir, 'docker-compose.yml')
        with open(docker_compose_file, 'w') as f:
            f.write(template)
        print(f"Created docker-compose.yml at: {docker_compose_file}")  # Debug log
        
        # Start user instance
        env = {
            'USER_ID': user_id,
            'NEO4J_PASSWORD': neo4j_password,
            'NEO4J_HTTP_PORT': str(neo4j_http_port),
            'NEO4J_BOLT_PORT': str(neo4j_bolt_port),
            'GEMINI_API_KEY': API_KEYS['GEMINI_API_KEY'],
            'GROQ_API_KEY': API_KEYS['GROQ_API_KEY'],
            'PORT': str(app_port)
        }
        
        # Store user information
        users[user_id] = {
            'created_at': time.time(),
            'last_active': time.time(),
            'neo4j_http_port': neo4j_http_port,
            'neo4j_bolt_port': neo4j_bolt_port,
            'app_port': app_port,
            'status': 'starting'
        }
        save_users(users)
        
        # Start Docker Compose
        try:
            cmd = f"docker compose -f {docker_compose_file} up -d"
            print(f"Executing command: {cmd} in directory: {user_dir}")  # Debug log
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=user_dir,
                env={**os.environ, **env},
                capture_output=True,
                text=True
            )
            print(f"Command output: {result.stdout}")  # Debug log
            print(f"Command error: {result.stderr}")  # Debug log
            result.check_returncode()  # Raise exception if command failed
            
            users[user_id]['status'] = 'running'
            save_users(users)
            
            return jsonify({
                'user_id': user_id,
                'app_url': f'http://localhost:{app_port}',
                'neo4j_url': f'http://localhost:{neo4j_http_port}',
                'status': 'created'
            })
        except subprocess.CalledProcessError as e:
            users[user_id]['status'] = 'failed'
            save_users(users)
            error_msg = f'Failed to start instance: {str(e)}\nOutput: {e.stdout}\nError: {e.stderr}'
            print(error_msg)  # Print to container logs
            return jsonify({
                'error': error_msg,
                'user_id': user_id
            }), 500
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error creating instance: {str(e)}\n{error_detail}")
        return jsonify({'error': f'Error creating instance: {str(e)}'}), 500

        
@app.route('/stop_instance/<user_id>', methods=['POST'])
def stop_instance(user_id):
    """Stop a user instance"""
    users = load_users()
    
    if user_id not in users:
        return jsonify({'error': 'User not found'}), 404
    
    user_dir = os.path.join(INSTANCES_DIR, user_id)
    
    if not os.path.exists(user_dir):
        users[user_id]['status'] = 'missing'
        save_users(users)
        return jsonify({'error': 'Instance directory not found'}), 404
    
    try:
        cmd = ["docker", "compose", "-f", os.path.join(user_dir, "docker-compose.yml"), "down"]
        subprocess.run(cmd, cwd=user_dir, check=True)
        users[user_id]['status'] = 'stopped'
        save_users(users)
        return jsonify({'status': 'stopped', 'user_id': user_id})
    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': f'Failed to stop instance: {str(e)}',
            'user_id': user_id
        }), 500

@app.route('/delete_instance/<user_id>', methods=['DELETE'])
def delete_instance(user_id):
    """Delete a user instance"""
    users = load_users()
    
    if user_id not in users:
        return jsonify({'error': 'User not found'}), 404
    
    user_dir = os.path.join(INSTANCES_DIR, user_id)
    
    if os.path.exists(user_dir):
        try:
            # Stop containers first
            cmd = f"cd {user_dir} && docker compose down"
            subprocess.run(cmd, shell=True, check=True)
            
            # Remove directory
            shutil.rmtree(user_dir)
        except Exception as e:
            return jsonify({
                'error': f'Failed to delete instance: {str(e)}',
                'user_id': user_id
            }), 500
    
    # Remove user from database
    if user_id in users:
        del users[user_id]
        save_users(users)
    
    return jsonify({'status': 'deleted', 'user_id': user_id})

@app.route('/list_instances', methods=['GET'])
def list_instances():
    """List all user instances"""
    users = load_users()
    return jsonify({'instances': users})

def find_available_port(start_port):
    """Find an available port starting from start_port"""
    # In a production environment, you would check if the port is actually available
    # For simplicity, we'll just increment based on existing users
    users = load_users()
    used_ports = set()
    
    for user_info in users.values():
        used_ports.add(user_info.get('neo4j_http_port', 0))
        used_ports.add(user_info.get('neo4j_bolt_port', 0))
        used_ports.add(user_info.get('app_port', 0))
    
    port = start_port
    while port in used_ports:
        port += 1
    
    return port

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
