def main():
    print("Hello from rag-chatbot!")


if __name__ == "__main__":
    main()

'''RAG CHATBOT That ingest Nipah Virus guidelines documents 
    1. parse.py -> takes corpus files and reproduce them in markdown and text format.
    2. build_faiss_store.py -> takes markdown files and build FAISS Vector store

    uv add llama-index llama-index-embeddings-huggingface llama-index-vector-stores-faiss faiss-cpu
    uv run parse.py
    uv run build_faiss_store.py
    

'''