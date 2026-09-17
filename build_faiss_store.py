"""
Build a FAISS vector store from the 'Preprocessed Files' directory (.md files only),
using LlamaIndex so chunking respects the markdown structure (headers first, then
size-bounded splits within each section) instead of a blind character window.

If a store already exists at the output path, the script skips ingestion entirely.

Install:
    pip install llama-index llama-index-embeddings-huggingface \
                llama-index-vector-stores-faiss faiss-cpu

Usage:
    python build_faiss_store.py
    python build_faiss_store.py --corpus "Preprocessed Files" --out ./vector_store
    python build_faiss_store.py --force   # rebuild even if a store already exists
"""
import argparse
import sys
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def store_exists(out_dir: Path) -> bool:
    # A persisted LlamaIndex/FAISS store is a directory of several files
    # (docstore.json, index_store.json, the raw faiss index, ...). Rather than
    # hard-coding filenames that shift between versions, just check the dir
    # exists and isn't empty.
    return out_dir.exists() and any(out_dir.iterdir())


def load_documents(corpus_dir: Path):
    # required_exts restricts SimpleDirectoryReader to .md only, ignoring
    # anything else that might be sitting in the folder.
    reader = SimpleDirectoryReader(input_dir=str(corpus_dir), required_exts=[".md"], recursive=False)
    return reader.load_data()


def build_index(docs, out_dir: Path) -> VectorStoreIndex:
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Split on markdown headers first so a chunk never straddles two unrelated
    # sections, then cap oversized sections at ~512 tokens with overlap.
    Settings.transformations = [
        MarkdownNodeParser(),
        SentenceSplitter(chunk_size=512, chunk_overlap=64),
    ]

    dim = len(Settings.embed_model.get_text_embedding("probe"))
    faiss_index = faiss.IndexFlatIP(dim)  # inner product; embeddings are pre-normalized
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir=str(out_dir))
    return index


def load_existing_index(out_dir: Path) -> VectorStoreIndex:
    from llama_index.vector_stores.faiss import FaissVectorStore

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FaissVectorStore.from_persist_dir(str(out_dir))
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(out_dir))
    return load_index_from_storage(storage_context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="Preprocessed Files", help="Directory of .md files")
    parser.add_argument("--out", default="vector_store", help="Directory the FAISS store lives in")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a store already exists")
    args = parser.parse_args()

    out_dir = Path(args.out)

    # Check for an existing store FIRST — before touching the corpus or loading a model.
    if store_exists(out_dir) and not args.force:
        print(f"Vector store already exists at {out_dir} — loading it instead of rebuilding.")
        load_existing_index(out_dir)
        print("Loaded. Use --force to rebuild from the corpus instead.")
        sys.exit(0)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        sys.exit(f"Corpus directory not found: {corpus_dir}")

    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files found in {corpus_dir}")

    docs = load_documents(corpus_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_index(docs, out_dir)
    print(f"Indexed {len(md_files)} file(s) into {out_dir}")


if __name__ == "__main__":
    main()
