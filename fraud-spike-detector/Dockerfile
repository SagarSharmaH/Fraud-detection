# Dockerfile for SentinelRisk-AI Engine & Dashboard
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Run full pipeline to generate initial synthetic dataset, train models, and save artifacts
RUN python run_all.py

# Expose FastAPI (8000) and Streamlit Dashboard (8501) ports
EXPOSE 8000 8501

# Default command launches Streamlit Dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
