# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode
# This ensures Python can find the trader module
RUN pip install -e .

# Create data directory
RUN mkdir -p data

# Set Python path explicitly
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "-m", "trader", "--help"]