# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the necessary application files
COPY --from=builder /app/api.py /app/
COPY --from=builder /app/ImageProcessor.py /app/
COPY --from=builder /app/templates/ /app/templates/
COPY --from=builder /app/drowsiness/ /app/drowsiness/
COPY --from=builder /app/Model/ /app/Model/
COPY --from=builder /app/Align/ /app/Align/
COPY --from=builder /app/Landmark/ /app/Landmark/
COPY --from=builder /app/Verify/ /app/Verify/
COPY --from=builder /app/UsersDatabaseHandeler/ /app/UsersDatabaseHandeler/
COPY --from=builder /app/ImageUtilis/ /app/ImageUtilis/
COPY --from=builder /app/CameraUtilis/ /app/CameraUtilis/

# Set environment variables for display and audio
ENV DISPLAY=:0
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native
ENV XDG_RUNTIME_DIR=/run/user/1000

# Expose the Flask port
EXPOSE 9000

# Command to run the application
ENTRYPOINT ["python", "api.py"]
