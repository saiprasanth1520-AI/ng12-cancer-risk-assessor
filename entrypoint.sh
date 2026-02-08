#!/bin/bash
set -e

# Run PDF ingestion if the vector store does not exist yet
if [ ! -d "/app/chroma_db" ] || [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
    echo "========================================"
    echo "Vector store not found — running PDF ingestion..."
    echo "========================================"
    python scripts/ingest_pdf.py
fi

echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
