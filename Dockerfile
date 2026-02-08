# ── Stage 1: Builder ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a prefix we can copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create directories for volumes (permissions persist into named volumes on first mount)
RUN mkdir -p /app/chroma_db /app/data

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

# Run as non-root user
USER appuser

# On first run the entrypoint will ingest the PDF and build the vector store
ENTRYPOINT ["./entrypoint.sh"]
