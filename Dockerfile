# DiagnostiCAI - Production Dockerfile
# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data, models, and results
RUN mkdir -p /app/dataset /app/models /app/results

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Run app5.py by default (most feature-complete)
# Can be overridden at runtime: docker run -e APP_FILE=app4.py ...
ENV APP_FILE=app5.py

# Start Streamlit application
CMD streamlit run ${APP_FILE} \
    --server.port=${STREAMLIT_SERVER_PORT} \
    --server.address=${STREAMLIT_SERVER_ADDRESS} \
    --server.headless=${STREAMLIT_SERVER_HEADLESS}
