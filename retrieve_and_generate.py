"""
Retrieve from the LlamaIndex/FAISS store built by build_faiss_store.py, then
generate an answer that cites its top-k sources. Config lives in config.py.

Install:
    pip install llama-index llama-index-embeddings-huggingface \
                llama-index-vector-stores-faiss faiss-cpu \
                python-dotenv requests

.env:
    OPENROUTER_API_KEY=...

Usage:
    python retrieve_and_generate.py "What is the incubation period of Nipah virus?"
    python retrieve_and_generate.py "..." --store ./vector_store --top-k 3
"""
import argparse
import json
from pathlib import Path

import requests

from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, STORE_DIR, TOP_K
from evaluate import evaluate_generation
from retrieval import load_retriever


def build_prompt(query: str, nodes) -> str:
    numbered_context = "\n\n".join(
        f"[{i + 1}] (source: {n.node.metadata.get('file_name', 'unknown')})\n{n.node.get_content()}"
        for i, n in enumerate(nodes)
    )
    return f"""Answer the question using only the numbered context below. If the answer isn't in the context, say so.
When you use a piece of context, cite it inline with its bracket number, e.g. [1].

Context:
{numbered_context}

Question: {query}
Answer:"""


def generate(prompt: str) -> str:
    resp = requests.post(
        url=f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": True},
        }),
    )
    data = resp.json()
    if "choices" not in data:
        raise Exception(f"Chat completion API error: {data}")
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--store", default=str(STORE_DIR), help="Directory the FAISS store lives in")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    retriever = load_retriever(Path(args.store), args.top_k)
    nodes = retriever.retrieve(args.query)

    prompt = build_prompt(args.query, nodes)
    answer = generate(prompt)

    print("Response:")
    print(answer)
    print("\nSources:")
    for i, n in enumerate(nodes):
        source = n.node.metadata.get("file_name", "unknown")
        score = f"{n.score:.3f}" if n.score is not None else "n/a"
        print(f"[{i + 1}] {source}  (score: {score})")

    context_chunks = [n.node.get_content() for n in nodes]
    report = evaluate_generation(args.query, context_chunks, answer)

    print("\nEvaluation:")
    print(f"  Faithfulness:      {report['faithfulness_score']:.2f}")
    print(f"  Hallucination rate: {report['hallucination_rate']:.2f}")
    if report["unsupported_claims"]:
        print("  Unsupported claims:")
        for claim in report["unsupported_claims"]:
            print(f"    - {claim}")
    print(f"  Answer relevancy:  {report['relevancy_score']:.2f} — {report['relevancy_explanation']}")


if __name__ == "__main__":
    main()
