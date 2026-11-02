"""
LLM-as-judge evaluation suite for the Nipah RAG chatbot.

Metrics implemented:
  - Faithfulness      (generation-side): fraction of claims in the answer
                        directly supported by retrieved context. No ground
                        truth needed -> hallucination_rate = 1 - faithfulness.
  - Answer Relevancy   (generation-side): does the answer address the
                        question asked, independent of factual correctness?
                        No ground truth needed.
  - Context Relevance  (retrieval-side): of the retrieved chunks, how many
                        are actually relevant to the question? Distinct from
                        faithfulness — this scores what was RETRIEVED, not
                        what the model SAID. No ground truth needed.
  - Answer Correctness (generation-side, needs a reference answer): hybrid
                        of embedding similarity (catches paraphrase) and an
                        LLM judge (catches factual contradiction that
                        similarity alone misses).

Install:
    pip install requests python-dotenv numpy

.env:
    OPENROUTER_API_KEY=...

Usage (as a library):
    from evaluate import evaluate_generation, evaluate_retrieval, score_answer_correctness
    report = evaluate_generation(query, context_chunks, answer)
    ctx_report = evaluate_retrieval(query, context_chunks)
    correctness = score_answer_correctness(answer, reference_answer)
"""
import json
import os
import re

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = "upstage/solar-pro4"  # swap for a stronger model if you want a more reliable judge
EMBED_MODEL = "openai/text-embedding-3-small"


def _call_judge(prompt: str, retries: int = 1) -> dict | None:
    """Returns the parsed judge response, or None if the judge failed to
    return valid JSON after retries — callers must handle None rather than
    assume keys exist, since a judge call can fail independently of the
    generation call it's scoring."""
    for attempt in range(retries + 1):
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning": {"enabled": False},
            }),
        )
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"Judge API error: {data}")
        raw = data["choices"][0]["message"]["content"]
        # Judge prompts ask for JSON only, but strip code fences defensively
        # in case the model wraps its output in ```json anyway.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt < retries:
                continue
            print(f"WARNING: judge returned non-JSON after {retries + 1} attempt(s), skipping this score.\nRaw response: {raw[:300]}")
            return None


def _embed(text: str) -> np.ndarray:
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
    )
    data = resp.json()
    if "data" not in data:
        raise Exception(f"Embedding API error: {data}")
    return np.array(data["data"][0]["embedding"])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Generation-side: Faithfulness
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, context_chunks: list[str]) -> dict:
    """Fraction of claims in `answer` that are directly supported by
    `context_chunks`. hallucination_rate = 1 - faithfulness_score."""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a strict fact-checker. Break the ANSWER into individual factual claims,
then decide for each claim whether it is directly supported by the CONTEXT.
A claim is only "supported" if the CONTEXT states it or clearly implies it — not
if it merely sounds plausible.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY this JSON, no other text:
{{
  "claims": [{{"claim": "...", "supported": true}}],
  "faithfulness_score": 0.0
}}
faithfulness_score is (# supported claims) / (total claims), between 0.0 and 1.0."""
    result = _call_judge(prompt)
    if result is None:
        return {"claims": [], "faithfulness_score": None, "hallucination_rate": None}
    result["hallucination_rate"] = round(1 - result["faithfulness_score"], 3)
    return result


# ---------------------------------------------------------------------------
# Generation-side: Answer Relevancy
# ---------------------------------------------------------------------------

def score_relevancy(query: str, answer: str) -> dict:
    """How directly the answer addresses the question, independent of whether
    it's factually correct."""
    prompt = f"""Rate how directly the ANSWER addresses the QUESTION, on a scale of
0.0 (irrelevant / non-answer) to 1.0 (fully and directly addresses it).

QUESTION: {query}
ANSWER: {answer}

Respond with ONLY this JSON, no other text:
{{"relevancy_score": 0.0, "explanation": "..."}}"""
    result = _call_judge(prompt)
    if result is None:
        return {"relevancy_score": None, "explanation": "judge call failed to return valid JSON"}
    return result


# ---------------------------------------------------------------------------
# Retrieval-side: Context Relevance
# ---------------------------------------------------------------------------

def score_context_relevance(query: str, context_chunks: list[str]) -> dict:
    """Of the retrieved chunks, what fraction are actually relevant to the
    question? Distinct from faithfulness: this scores what retrieval
    returned, before generation ever happens, so a low score here points at
    the retriever/chunking/embedding, not the LLM's answer."""
    numbered_chunks = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    prompt = f"""You are evaluating a retrieval system. For each numbered CHUNK below, decide
whether it is relevant to answering the QUESTION — i.e. whether it contains
information that would help answer it, even partially.

QUESTION: {query}

CHUNKS:
{numbered_chunks}

Respond with ONLY this JSON, no other text:
{{
  "chunk_relevance": [{{"chunk_index": 1, "relevant": true}}],
  "context_relevance_score": 0.0
}}
context_relevance_score is (# relevant chunks) / (total chunks), between 0.0 and 1.0.
Include one entry in chunk_relevance for every chunk, in order."""
    result = _call_judge(prompt)
    if result is None:
        return {"chunk_relevance": [], "context_relevance_score": None}
    return result


# ---------------------------------------------------------------------------
# Generation-side: Answer Correctness (needs a reference answer)
# ---------------------------------------------------------------------------

def score_answer_correctness(answer: str, reference_answer: str) -> dict:
    """Hybrid metric: embedding similarity catches paraphrase (different
    wording, same meaning), the LLM judge catches factual contradiction that
    similarity alone misses (e.g. "5 members" vs "4 members" can still
    embed as similar sentences despite being factually wrong).
    Final score is the average of the two — deliberately conservative,
    since either signal alone can be misleading on its own."""
    sim = _cosine_similarity(_embed(answer), _embed(reference_answer))

    prompt = f"""Compare the GENERATED ANSWER to the REFERENCE ANSWER. Rate factual correctness
on a scale of 0.0 (contradicts or misses the key facts) to 1.0 (fully consistent
with the reference, even if worded differently).

REFERENCE ANSWER: {reference_answer}
GENERATED ANSWER: {answer}

Respond with ONLY this JSON, no other text:
{{"correctness_score": 0.0, "explanation": "..."}}"""
    judge_result = _call_judge(prompt)

    if judge_result is None:
        return {
            "embedding_similarity": round(sim, 3),
            "judge_correctness_score": None,
            "judge_explanation": "judge call failed to return valid JSON",
            "answer_correctness_score": None,
        }

    return {
        "embedding_similarity": round(sim, 3),
        "judge_correctness_score": judge_result["correctness_score"],
        "judge_explanation": judge_result["explanation"],
        "answer_correctness_score": round((sim + judge_result["correctness_score"]) / 2, 3),
    }


# ---------------------------------------------------------------------------
# Combined reports
# ---------------------------------------------------------------------------

def evaluate_generation(query: str, context_chunks: list[str], answer: str) -> dict:
    """Faithfulness + relevancy. No ground truth required — runs on any query."""
    faithfulness = score_faithfulness(answer, context_chunks)
    relevancy = score_relevancy(query, answer)
    return {
        "faithfulness_score": faithfulness["faithfulness_score"],
        "hallucination_rate": faithfulness["hallucination_rate"],
        "unsupported_claims": [c["claim"] for c in faithfulness["claims"] if not c["supported"]],
        "relevancy_score": relevancy["relevancy_score"],
        "relevancy_explanation": relevancy["explanation"],
    }


def evaluate_retrieval(query: str, context_chunks: list[str]) -> dict:
    """Context relevance only. No ground truth required — runs on any query."""
    result = score_context_relevance(query, context_chunks)
    return {
        "context_relevance_score": result["context_relevance_score"],
        "chunk_relevance": result["chunk_relevance"],
    }


def evaluate_full(query: str, context_chunks: list[str], answer: str, reference_answer: str | None = None) -> dict:
    """Runs every metric available. Answer correctness only runs if a
    reference_answer is supplied, since it needs ground truth."""
    report = {
        **evaluate_retrieval(query, context_chunks),
        **evaluate_generation(query, context_chunks, answer),
    }
    if reference_answer:
        report["answer_correctness"] = score_answer_correctness(answer, reference_answer)
    return report


if __name__ == "__main__":
    # Quick manual smoke test
    ctx = ["Nipah virus was first identified in Malaysia in 1998-1999 among pig farmers."]
    ans = "Nipah virus was first identified in Malaysia in 1998, and it originated in bats in Australia."
    ref = "Nipah virus was first identified in 1998-1999 in Malaysia among pig farmers."

    print("--- Retrieval eval ---")
    print(json.dumps(evaluate_retrieval("When and where was Nipah first identified?", ctx), indent=2))

    print("\n--- Generation eval ---")
    print(json.dumps(evaluate_generation("When and where was Nipah first identified?", ctx, ans), indent=2))

    print("\n--- Answer correctness ---")
    print(json.dumps(score_answer_correctness(ans, ref), indent=2))
