FROM python:3.10.16-slim

# Set working directory
WORKDIR /app

# Install only the essential system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
# RUN python3.10 -m pip install --no-cache-dir --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .