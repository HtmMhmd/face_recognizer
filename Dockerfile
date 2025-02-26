FROM python:3.10.16-slim

# Prevent interactive prompts during package installation
# ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pulseaudio pulseaudio-utils libasound2-plugins alsa-utils \
    && rm -rf /var/lib/apt/lists/*


# Set PulseAudio environment variables
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native
ENV XDG_RUNTIME_DIR=/run/user/1000

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

# Copy all application files
COPY . .

# Copy alarm file
COPY alarm2.mp3 /app/alarm.mp3

# # Set the default command
# CMD ["python3.10", "main.py"]