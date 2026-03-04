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

# Cache bust: change this value to force full rebuild on Render
ARG CACHEBUST=2026030401

# Copy installed packages from builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source code
COPY app/ ./app/
COPY src/ ./src/

# Copy pre-trained model artifacts
COPY model/ ./model/

# Download CSV data files from GitHub (Render build context excludes large files)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
  mkdir -p ./data/raw && \
  curl -fsSL "https://raw.githubusercontent.com/brunoymo/tech_challenge_5/main/backend/data/raw/dataset_2022.csv" -o ./data/raw/dataset_2022.csv && \
  curl -fsSL "https://raw.githubusercontent.com/brunoymo/tech_challenge_5/main/backend/data/raw/dataset_2023.csv" -o ./data/raw/dataset_2023.csv && \
  curl -fsSL "https://raw.githubusercontent.com/brunoymo/tech_challenge_5/main/backend/data/raw/dataset_2024.csv" -o ./data/raw/dataset_2024.csv && \
  apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && \
  ls -la ./data/raw/



# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Health check so Docker (or Compose) knows when the API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
