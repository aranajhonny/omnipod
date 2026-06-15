"""Configuration and constants for OmniPod."""

import os

from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
LLM_MODEL = "deepseek-v4-flash"

# ── Qdrant ───────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "podcasts_hybrid")

# ── Embedding ────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ── Retrieval ────────────────────────────────────────────────────
HYBRID_TOP_K = 15
RERANKER_TOP_K = 5
SUB_QUERIES_COUNT = 3

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "lex_podcast", "data", "transcripts")
