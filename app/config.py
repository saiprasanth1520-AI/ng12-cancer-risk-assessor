import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider: "vertex_ai" (GCP — default) or "google_genai" (API key)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex_ai")

# Google AI Studio
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Vertex AI (optional)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# RAG
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Redis (for chat session memory)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "86400"))  # 24 hours

# Hybrid search / reranking
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "1.0"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "1.0"))
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "10"))

# Auth & rate limiting
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_ASSESS = os.getenv("RATE_LIMIT_ASSESS", "10/minute")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "30/minute")

# Agent
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "20"))

# Corrective RAG
CORRECTIVE_RAG_ENABLED = os.getenv("CORRECTIVE_RAG_ENABLED", "true").lower() == "true"
CORRECTIVE_RAG_THRESHOLD = float(os.getenv("CORRECTIVE_RAG_THRESHOLD", "0.3"))
CORRECTIVE_RAG_MAX_RETRIES = int(os.getenv("CORRECTIVE_RAG_MAX_RETRIES", "2"))

# Timeouts
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
RAG_TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "30"))

# LLM Judge
LLM_JUDGE_ENABLED = os.getenv("LLM_JUDGE_ENABLED", "true").lower() == "true"
LLM_JUDGE_TIMEOUT = int(os.getenv("LLM_JUDGE_TIMEOUT", "60"))
LLM_JUDGE_CROSS_EXAMINE = os.getenv("LLM_JUDGE_CROSS_EXAMINE", "false").lower() == "true"

# Structured logging
STRUCTURED_LOGGING_ENABLED = os.getenv("STRUCTURED_LOGGING_ENABLED", "false").lower() == "true"

# Data paths
PATIENTS_JSON_PATH = os.getenv("PATIENTS_JSON_PATH", "./data/patients.json")
NG12_PDF_PATH = os.getenv("NG12_PDF_PATH", "./data/ng12_guidelines.pdf")

# Metadata filtering
METADATA_FILTER_ENABLED = os.getenv("METADATA_FILTER_ENABLED", "true").lower() == "true"

# Cache warm-up
CACHE_WARMUP_ENABLED = os.getenv("CACHE_WARMUP_ENABLED", "true").lower() == "true"
