# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first and install dependencies
COPY myenv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path to include current directory
ENV PYTHONPATH=/app
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Copy load_model.py and the model file first
COPY myenv/load_model.py .
COPY myenv/small640.pt .

# Pre-load and verify the model during build
RUN python3 -c "from load_model import InferenceModel; print('Preloading model...'); model = InferenceModel('small640.pt'); print('Model loaded successfully')"

# Copy remaining application files
COPY myenv/main.py .
COPY myenv/service.py .
COPY myenv/load_video.py .
COPY myenv/model.py .

# Create required directories
RUN mkdir -p uploads recordings

# Run the application with increased timeout
CMD exec gunicorn --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --graceful-timeout 300 \
    --keep-alive 120 \
    --log-level debug \
    main:app
