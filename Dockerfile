FROM python:3.10

# Prevent interactive prompts during package installation
# ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN python3.10 -m pip install --no-cache-dir --upgrade pip

# # Install the required Python packages
# RUN pip install --no-cache-dir \
#     opencv-python-headless \
#     tflite-runtime \
#     mediapipe \
#     numpy

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .


# # Set the default command
# CMD ["python3.10", "main.py"]