"""Shared index loading, so the CLI, the retrieval eval and the full eval all
retrieve identically (previously load_retriever was copy-pasted into three files)."""
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from config import EMBED_MODEL_NAME, STORE_DIR, TOP_K


def set_embed_model():
    # Must match the model the store was built with, or query vectors won't
    # line up with the indexed ones.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def load_index(store_dir: Path = STORE_DIR):
    store_dir = Path(store_dir)
    if not store_dir.exists() or not any(store_dir.iterdir()):
        raise SystemExit(f"No vector store found at {store_dir}. Run build_faiss_store.py first.")
    set_embed_model()
    vector_store = FaissVectorStore.from_persist_dir(str(store_dir))
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(store_dir))
    return load_index_from_storage(storage_context)


def load_retriever(store_dir: Path = STORE_DIR, top_k: int = TOP_K):
    return load_index(store_dir).as_retriever(similarity_top_k=top_k)
