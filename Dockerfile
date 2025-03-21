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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY . .

# Stage 2: Runtime image using Chainguard Python
FROM python:3.10-alpine

WORKDIR /app

# Set up non-root user for better security
USER nonroot

# Install runtime dependencies for OpenCV
# Note: Chainguard images use apk for package management
# RUN apk --no-cache add \
#     libstdc++ \
#     libgcc \
#     mesa-gl \
#     glib \
#     pulseaudio

# Copy Python dependencies from builder
COPY --from=builder --chown=nonroot:nonroot /root/.local /home/nonroot/.local

# Set Python path to find the installed packages
ENV PATH=/home/nonroot/.local/bin:$PATH
ENV PYTHONPATH=/home/nonroot/.local/lib/python3.10/site-packages:$PYTHONPATH

# Copy only the necessary application files
COPY --from=builder --chown=nonroot:nonroot /app/api.py /app/
COPY --from=builder --chown=nonroot:nonroot /app/ImageProcessor.py /app/
COPY --from=builder --chown=nonroot:nonroot /app/templates/ /app/templates/
COPY --from=builder --chown=nonroot:nonroot /app/drowsiness/ /app/drowsiness/
COPY --from=builder --chown=nonroot:nonroot /app/Model/ /app/Model/
COPY --from=builder --chown=nonroot:nonroot /app/Align/ /app/Align/
COPY --from=builder --chown=nonroot:nonroot /app/Landmark/ /app/Landmark/
COPY --from=builder --chown=nonroot:nonroot /app/Verify/ /app/Verify/
COPY --from=builder --chown=nonroot:nonroot /app/UsersDatabaseHandeler/ /app/UsersDatabaseHandeler/
COPY --from=builder --chown=nonroot:nonroot /app/ImageUtilis/ /app/ImageUtilis/
COPY --from=builder --chown=nonroot:nonroot /app/CameraUtilis/ /app/CameraUtilis/

# Set environment variables for display and audio
ENV DISPLAY=:0
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native
ENV XDG_RUNTIME_DIR=/run/user/1000

# Expose the Flask port
EXPOSE 9000

# Command to run the application
CMD ["python", "api.py"]
