# Dockerfile for SentinelRisk-AI Engine & Dashboard
# Multi-stage build for smaller production image

# --- Stage 1: Build & Train ---
FROM python:3.11-slim AS builder

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

# --- Stage 2: Production Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r sentinelrisk && useradd -r -g sentinelrisk sentinelrisk

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and generated artifacts
COPY --from=builder /app/src ./src
COPY --from=builder /app/data ./data
COPY --from=builder /app/models ./models
COPY --from=builder /app/reports ./reports
COPY --from=builder /app/run_all.py ./run_all.py
COPY --from=builder /app/requirements.txt ./requirements.txt
COPY --from=builder /app/.streamlit ./.streamlit

# Switch to non-root user
USER sentinelrisk

# Expose FastAPI (8000) and Streamlit Dashboard (8501) ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command launches Streamlit Dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
