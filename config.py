"""Single source of truth for paths, models and retrieval settings.

Everything that was previously hard-coded across build_faiss_store.py,
retrieve_and_generate.py, evaluate.py and the eval runners lives here, so an
experiment changes one value in one place.

Environment (.env, see .env.example):
    OPENROUTER_API_KEY   required for generation and LLM-as-judge scoring
    OPENROUTER_MODEL     generator model id
    LLAMA_CLOUD_API_KEY  only needed by parse.py (PDF -> markdown)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).parent
PDF_DIR = ROOT / "corpus"
CORPUS_DIR = ROOT / "Preprocessed Files"   # parsed .md files the index is built from
STORE_DIR = ROOT / "vector_store"
RESULTS_DIR = ROOT / "results"

# --- Models ----------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # local; index and queries MUST use the same one
JUDGE_MODEL = "upstage/solar-pro4"
JUDGE_EMBED_MODEL = "openai/text-embedding-3-small"   # used only by answer-correctness similarity

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Accept the old hyphenated name so existing .env files keep working.
LLM_MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENROUTER-MODEL")

# --- Chunking / retrieval --------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 3
