# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install minimal build dependencies for OpenCV headless
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with opencv-headless instead of full opencv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install minimal runtime dependencies for OpenCV headless
RUN apt-get update && apt-get install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the Flask port
EXPOSE 9000

# Command to run the application
# ENTRYPOINT ["python", "api.py"]
