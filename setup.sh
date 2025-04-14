#!/bin/bash

# Create required directories
mkdir -p user_instances
mkdir -p html

# Copy HTML template
cp html/index.html html/

# Create .env file
if [ ! -f .env ]; then
    echo "Creating .env file"
    echo "GEMINI_API_KEY=your_gemini_api_key" > .env
    echo "GROQ_API_KEY=your_groq_api_key" >> .env
    echo "Please edit .env file and add your API keys"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

# Start the main services
echo "Starting main services..."
docker-compose up -d

echo "Setup complete! The system should be available at http://localhost"
