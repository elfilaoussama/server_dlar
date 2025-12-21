# =============================================================================
# Document Layout Analysis & Reconstruction - Docker Image
# =============================================================================
# Multi-stage build for optimized image size
#
# Build: docker build -t dla-api .
# Run:   docker run -p 8000:8000 dla-api
# =============================================================================

FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Final Stage
# =============================================================================
FROM python:3.11-slim

# Build arguments for model/data download URLs
# Override these at build time if needed:
#   docker build --build-arg YOLO_MODEL_URL=<url> -t dla-api .
ARG YOLO_MODEL_URL="https://drive.google.com/uc?export=download&id=1cBQZ3FvEX-bebqJyLgZVqzl6jmSN7ccT"
ARG TESSDATA_ENG_URL="https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
ARG TESSDATA_FRA_URL="https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata"
ARG TESSDATA_ARA_URL="https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata"

# Labels
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Document Layout Analysis & Reconstruction API"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    APP_NAME="Document Layout Analysis API" \
    APP_VERSION="1.0.0" \
    ENVIRONMENT="production" \
    DEBUG="false" \
    HOST="0.0.0.0" \
    PORT="8000" \
    # Tesseract
    TESSERACT_CMD="/usr/bin/tesseract" \
    TESSDATA_PREFIX="/app/tessdata" \
    # Logging
    LOG_LEVEL="INFO" \
    LOG_FORMAT="json" \
    LOG_FILE_ENABLED="true" \
    LOG_FILE_PATH="/app/logs/app.log"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR
    tesseract-ocr \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # PDF processing (for pdf2image)
    poppler-utils \
    # Download tools
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user (non-root)
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create directories
RUN mkdir -p /app/models /app/tessdata /app/logs /app/output && \
    chown -R appuser:appuser /app

# Download Tesseract language data
RUN echo "Downloading Tesseract language data..." && \
    wget -q -O /app/tessdata/eng.traineddata "${TESSDATA_ENG_URL}" && \
    wget -q -O /app/tessdata/fra.traineddata "${TESSDATA_FRA_URL}" && \
    wget -q -O /app/tessdata/ara.traineddata "${TESSDATA_ARA_URL}" && \
    echo "Tesseract data downloaded successfully"

# Download YOLO model (using gdown for Google Drive)
# Note: Replace YOUR_YOLO_MODEL_FILE_ID with actual Google Drive file ID
RUN pip install --no-cache-dir gdown && \
    echo "Downloading YOLO model..." && \
    gdown --fuzzy "${YOLO_MODEL_URL}" -O /app/models/yolov10m_doclaynet.onnx || \
    echo "Warning: YOLO model download failed. Please provide model manually." && \
    pip uninstall -y gdown

# Copy application code
COPY --chown=appuser:appuser . /app/

# Ensure directories exist with correct permissions (after COPY)
RUN mkdir -p /app/logs /app/output && chown -R appuser:appuser /app/logs /app/output

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# =============================================================================
# Production: Gunicorn with Uvicorn workers
# =============================================================================
# Benefits over plain Uvicorn:
# - Process management (auto-restart crashed workers)
# - Graceful worker restarts
# - Pre-fork model for better resource usage
# - Signal handling (SIGTERM, SIGHUP)
# - Multiple worker processes for parallelism
#
# Workers formula: (2 * CPU cores) + 1
# For 4 cores: 9 workers
# =============================================================================

# Install Gunicorn
RUN pip install --no-cache-dir gunicorn

# Default command: Gunicorn with Uvicorn workers
# --workers: Number of worker processes (adjust based on CPU cores)
# --worker-class: Use Uvicorn's worker class for ASGI
# --timeout: Worker timeout (increase for long-running AI tasks)
# --keep-alive: Keep-alive timeout for connections
# --access-logfile: Log requests (- for stdout)
# --error-logfile: Log errors (- for stderr)
# --capture-output: Capture stdout/stderr from workers
CMD ["gunicorn", "server:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "--enable-stdio-inheritance"]
