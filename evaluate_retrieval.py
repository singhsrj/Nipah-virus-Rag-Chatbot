"""
Retrieval-side evaluation for the Nipah RAG chatbot: runs the labeled
`test_set` against the actual FAISS/LlamaIndex retriever built by
build_faiss_store.py, and scores it with precision@k, recall@k, MRR, and
hit rate.

This is deliberately separate from generation-side eval (see evaluate.py) —
a bad final answer can come from failed retrieval OR failed generation, and
conflating them hides which one to fix.

Usage:
    python build_faiss_store.py          # build the index once, if not already built
    python evaluate_retrieval.py         # run this
    python evaluate_retrieval.py --top-k 5
"""


import argparse
from pathlib import Path

from config import STORE_DIR, TOP_K
from retrieval import load_retriever
from test_set import test_set


def node_is_relevant(node, item):
    """A retrieved node counts as relevant if it comes from the labeled
    source_file AND (when snippets are given) contains at least one of the
    labeled snippets. Falling back to source_file alone handles cases where
    LlamaIndex's chunker splits a snippet across a chunk boundary."""
    file_name = node.node.metadata.get("file_name", "")
    if item["source_file"] is None:
        return False  # negative test case — nothing should count as relevant
    if item["source_file"] not in file_name:
        return False
    if not item["relevant_text_contains"]:
        return True  # no snippet given, source-file match is enough
    content = node.node.get_content()
    return any(snippet in content for snippet in item["relevant_text_contains"])


def precision_at_k(relevant_flags):
    if not relevant_flags:
        return 0.0
    return sum(relevant_flags) / len(relevant_flags)


def recall_at_least_one(relevant_flags, has_ground_truth):
    """Since we don't have exhaustive per-chunk ground truth (only 'does the
    labeled source appear'), recall here means: did at least one relevant
    chunk make it into top-k. For a corpus with a single labeled source doc
    per question, this collapses to hit rate — reported separately for
    clarity, but computed the same way at this granularity."""
    if not has_ground_truth:
        return None  # negative test case, recall is undefined
    return 1.0 if any(relevant_flags) else 0.0


def reciprocal_rank(relevant_flags):
    for rank, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            return 1 / rank
    return 0.0


def run_eval(top_k: int, store_dir: str):
    retriever = load_retriever(Path(store_dir), top_k)

    results = []
    for item in test_set:
        nodes = retriever.retrieve(item["question"])
        relevant_flags = [node_is_relevant(n, item) for n in nodes]

        has_ground_truth = item["source_file"] is not None
        p = precision_at_k(relevant_flags) if has_ground_truth else None
        hit = recall_at_least_one(relevant_flags, has_ground_truth)
        rr = reciprocal_rank(relevant_flags) if has_ground_truth else None

        results.append({
            "question": item["question"],
            "source_file": item["source_file"],
            "precision": p,
            "hit": hit,
            "rr": rr,
            "retrieved_sources": [n.node.metadata.get("file_name", "unknown") for n in nodes],
        })

        status = "NEGATIVE TEST" if item["source_file"] is None else ("HIT" if hit else "MISS")
        print(f"[{status}] {item['question']}")
        print(f"  retrieved from: {results[-1]['retrieved_sources']}")
        if has_ground_truth:
            print(f"  precision@{top_k}: {p:.2f}  RR: {rr:.2f}")
        print()

    scored = [r for r in results if r["source_file"] is not None]
    avg_precision = sum(r["precision"] for r in scored) / len(scored)
    hit_rate = sum(r["hit"] for r in scored) / len(scored)
    mrr = sum(r["rr"] for r in scored) / len(scored)

    negative_tests = [r for r in results if r["source_file"] is None]
    false_positive_rate = None
    if negative_tests:
        # For negative tests, "relevant" is undefined by construction, so we
        # instead report whether retrieval returned anything from a real
        # medical source at all vs. ideally flagging no-answer -- this is a
        # qualitative check, read the printed retrieved_sources by hand.
        false_positive_rate = "see printed output above — no automatic ground truth for negative cases"

    print("=== Retrieval Eval Aggregate ===")
    print(f"Questions scored: {len(scored)} (+ {len(negative_tests)} negative test case(s), reported separately)")
    print(f"Mean Precision@{top_k}: {avg_precision:.2f}")
    print(f"Hit Rate: {hit_rate:.2f}")
    print(f"MRR: {mrr:.2f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--store", default=str(STORE_DIR))
    args = parser.parse_args()

    run_eval(top_k=args.top_k, store_dir=args.store)
