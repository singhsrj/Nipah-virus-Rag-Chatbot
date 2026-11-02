"""
End-to-end evaluation runner for the Nipah RAG chatbot: for every question in
test_set.py, retrieves from the real FAISS index, generates an answer, and
scores it on all four metrics (context relevance, faithfulness, answer
relevancy, and — where a reference answer exists — answer correctness).

This is the script that produces the numbers for a project writeup / resume
bullet. Run it after build_faiss_store.py has built the index once.

Usage:
    python build_faiss_store.py
    python run_full_eval.py
    python run_full_eval.py --top-k 5 --out results.json
"""
import argparse
import json
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from evaluate import evaluate_full
from test_set import test_set
from reference_answers import reference_answers

# reuse generate() and build_prompt() from retrieve_and_generate.py so the
# eval hits the exact same generation path a real user query would
from retrieve_and_generate import build_prompt, generate


def load_retriever(store_dir: Path, top_k: int):
    if not store_dir.exists() or not any(store_dir.iterdir()):
        raise SystemExit(f"No vector store found at {store_dir}. Run build_faiss_store.py first.")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FaissVectorStore.from_persist_dir(str(store_dir))
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(store_dir))
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=top_k)


def run(top_k: int, store_dir: str, out_path: str | None):
    retriever = load_retriever(Path(store_dir), top_k)

    all_results = []
    for item in test_set:
        question = item["question"]
        is_negative_test = item["source_file"] is None

        nodes = retriever.retrieve(question)
        context_chunks = [n.node.get_content() for n in nodes]

        prompt = build_prompt(question, nodes)
        answer = generate(prompt)

        reference = reference_answers.get(question)
        report = evaluate_full(question, context_chunks, answer, reference_answer=reference)

        result = {
            "question": question,
            "is_negative_test": is_negative_test,
            "retrieved_sources": [n.node.metadata.get("file_name", "unknown") for n in nodes],
            "answer": answer,
            **report,
        }
        all_results.append(result)

        print(f"Q: {question}")
        print(f"  Sources: {result['retrieved_sources']}")
        print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"  Context Relevance: {report['context_relevance_score']:.2f}")
        print(f"  Faithfulness: {report['faithfulness_score']:.2f}  (hallucination rate: {report['hallucination_rate']:.2f})")
        print(f"  Relevancy: {report['relevancy_score']:.2f}")
        if "answer_correctness" in report:
            print(f"  Answer Correctness: {report['answer_correctness']['answer_correctness_score']:.2f}")
        if report.get("unsupported_claims"):
            print(f"  Unsupported claims: {report['unsupported_claims']}")
        print()

    # aggregate, excluding the deliberate negative test case, and excluding
    # any individual metric that failed to parse (None) from that metric's
    # own average rather than crashing or silently treating it as 0
    scored = [r for r in all_results if not r["is_negative_test"]]

    def safe_avg(values):
        clean = [v for v in values if v is not None]
        return (sum(clean) / len(clean), len(clean)) if clean else (None, 0)

    avg_context_relevance, n_cr = safe_avg([r["context_relevance_score"] for r in scored])
    avg_faithfulness, n_f = safe_avg([r["faithfulness_score"] for r in scored])
    avg_hallucination, n_h = safe_avg([r["hallucination_rate"] for r in scored])
    avg_relevancy, n_r = safe_avg([r["relevancy_score"] for r in scored])

    with_correctness = [r for r in scored if "answer_correctness" in r]
    avg_correctness, n_c = safe_avg([r["answer_correctness"]["answer_correctness_score"] for r in with_correctness])

    print("=== Aggregate (excluding negative test case) ===")
    print(f"Questions attempted: {len(scored)}")
    if avg_context_relevance is not None:
        print(f"Mean Context Relevance: {avg_context_relevance:.2f}  (n={n_cr})")
    if avg_faithfulness is not None:
        print(f"Mean Faithfulness: {avg_faithfulness:.2f}  (n={n_f})")
    if avg_hallucination is not None:
        print(f"Mean Hallucination Rate: {avg_hallucination:.2f}  (n={n_h})")
    if avg_relevancy is not None:
        print(f"Mean Answer Relevancy: {avg_relevancy:.2f}  (n={n_r})")
    if avg_correctness is not None:
        print(f"Mean Answer Correctness: {avg_correctness:.2f}  (n={n_c})")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--store", default="vector_store")
    parser.add_argument("--out", default=None, help="Optional path to write full JSON results")
    args = parser.parse_args()

    run(top_k=args.top_k, store_dir=args.store, out_path=args.out)
