# --- Stage 1: Builder ---
# Install dependencies in an isolated layer for smaller final image
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# --- Stage 2: Production ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source code
COPY app/ ./app/
COPY src/ ./src/

# Copy pre-trained model and data (must exist before building)
COPY model/ ./model/
COPY data/ ./data/

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Health check so Docker (or Compose) knows when the API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
