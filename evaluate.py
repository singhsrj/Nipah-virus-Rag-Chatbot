"""
LLM-as-judge evaluation: faithfulness (-> hallucination rate) and answer
relevancy, scored via the same OpenRouter chat-completions endpoint used for
generation. No ground-truth answers required, so this runs on any query.

Install:
    pip install requests python-dotenv

.env:
    OPENROUTER_API_KEY=...

Usage (as a library):
    from evaluate import evaluate_generation
    report = evaluate_generation(query, context_chunks, answer)
"""
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = "upstage/solar-pro4"  # swap for a stronger model if you want a more reliable judge


def _call_judge(prompt: str) -> dict:
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
    # Judge prompts ask for JSON only, but strip code fences defensively in
    # case the model wraps its output in ```json anyway.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


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
    result["hallucination_rate"] = round(1 - result["faithfulness_score"], 3)
    return result


def score_relevancy(query: str, answer: str) -> dict:
    """How directly the answer addresses the question, independent of whether
    it's factually correct."""
    prompt = f"""Rate how directly the ANSWER addresses the QUESTION, on a scale of
0.0 (irrelevant / non-answer) to 1.0 (fully and directly addresses it).

QUESTION: {query}
ANSWER: {answer}

Respond with ONLY this JSON, no other text:
{{"relevancy_score": 0.0, "explanation": "..."}}"""
    return _call_judge(prompt)


def evaluate_generation(query: str, context_chunks: list[str], answer: str) -> dict:
    faithfulness = score_faithfulness(answer, context_chunks)
    relevancy = score_relevancy(query, answer)
    return {
        "faithfulness_score": faithfulness["faithfulness_score"],
        "hallucination_rate": faithfulness["hallucination_rate"],
        "unsupported_claims": [c["claim"] for c in faithfulness["claims"] if not c["supported"]],
        "relevancy_score": relevancy["relevancy_score"],
        "relevancy_explanation": relevancy["explanation"],
    }


if __name__ == "__main__":
    # Quick manual smoke test
    ctx = ["Nipah virus was first identified in Malaysia in 1998-1999 among pig farmers."]
    ans = "Nipah virus was first identified in Malaysia in 1998, and it originated in bats in Australia."
    print(json.dumps(evaluate_generation("When and where was Nipah first identified?", ctx, ans), indent=2))
