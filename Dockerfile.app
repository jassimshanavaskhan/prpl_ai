# FROM python:3.9-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create directories for vector stores and uploads
# RUN mkdir -p vector_stores temp_uploads

# # Create credentials directory
# RUN mkdir -p /app/credentials

# # Expose the application port
# EXPOSE 5001

# # Start the application
# CMD ["python", "app2.py"]


FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app2.py .
COPY RDKAssistant_Class.py .
COPY neo4j_Class.py .
COPY VectorStoreManager.py .
COPY ContentGenerator.py .
COPY logger.py .

# Copy directories
COPY templates ./templates
COPY UNDER_TEST ./UNDER_TEST

# Create directory for vector stores
RUN mkdir -p vector_stores

# Expose the port
EXPOSE 5001

# Run the application
CMD ["python", "app2.py"]
