# 🦠 Nipah Virus RAG Chatbot

A retrieval-augmented generation chatbot over WHO and ECDC public-health guidance on Nipah virus — built with LlamaIndex and FAISS, and shipped with a **custom four-metric evaluation suite** that scores retrieval and generation quality independently using LLM-as-judge.

This isn't just a chatbot. It's a chatbot with a measurement layer that tells you *why* an answer was good or bad, at the chunk level and the claim level — not just a single opaque score.

---

## Why this project exists

Most RAG demos stop at "it answers questions." This one asks the harder question: **how do you know it's answering them well?** The evaluation suite here is the actual point of the project — it's what separates "I built a chatbot" from "I can tell you where a chatbot is failing and why."

---

## Architecture

```
Preprocessed Files/*.md          6 WHO/ECDC source documents (~23,500 words)
        │
        ▼
build_faiss_store.py             Markdown-aware chunking → FAISS vector index
        │                        (headers split first, then size-bounded
        │                         splits within each section)
        ▼
retrieve_and_generate.py         Query → top-k retrieval → cited answer
        │                        generation via OpenRouter
        ▼
evaluate.py                      4-metric LLM-as-judge evaluation suite
        │
        ▼
run_full_eval.py                 Orchestrates the full pipeline end-to-end
                                  against a hand-verified test set
```

| File | Role |
|---|---|
| `build_faiss_store.py` | Ingests the corpus, builds the FAISS index |
| `retrieve_and_generate.py` | Query-time retrieval + cited generation |
| `evaluate.py` | The 4 evaluation metrics, as reusable scorers |
| `test_set.py` | 20 hand-verified ground-truth questions across all 6 source documents |
| `reference_answers.py` | Hand-written reference answers, for the correctness metric |
| `evaluate_retrieval.py` | Retrieval-only evaluation (precision@k, hit rate, MRR) |
| `run_full_eval.py` | Full end-to-end run: retrieve → generate → score everything |

---

## The evaluation suite

Retrieval quality and generation quality are scored **separately** on purpose. A bad final answer can come from a bad retrieval *or* a bad generation — conflating the two into one score hides which one you actually need to fix.

| Metric | Question it answers | Ground truth needed? |
|---|---|:---:|
| **Context Relevance** | Of the chunks retrieved, how many are actually relevant? | No |
| **Faithfulness** | What fraction of claims in the answer are supported by retrieved context? | No |
| **Answer Relevancy** | Does the answer directly address the question asked? | No |
| **Answer Correctness** | Is the answer factually correct against a known-good reference? | Yes |

Faithfulness and Context Relevance both return **explainable** scores — every unsupported claim and every irrelevant chunk is logged individually, not folded into a single number you have to trust blindly.

```python
# Faithfulness scoring, conceptually
{
  "claims": [
    {"claim": "Nipah virus was first identified in Malaysia in 1998", "supported": true},
    {"claim": "it originated in bats in Australia", "supported": false}
  ],
  "faithfulness_score": 0.5
}
```

Answer Correctness is a deliberate hybrid: embedding similarity alone can be fooled by two sentences that are structurally similar but factually opposite ("the outbreak had 40% mortality" vs. "60% mortality" embed as nearly identical). The LLM judge half catches exactly that kind of contradiction that pure vector similarity misses.

---

## Results

Measured against 14 of 20 ground-truth questions from the first full evaluation run (see **[Known Issue](#known-issue--found-by-the-eval-suite-itself)** for why 6 didn't complete):

<table>
<tr><td><b>Faithfulness</b></td><td>0.94</td></tr>
<tr><td><b>Hallucination Rate</b></td><td>0.06</td></tr>
<tr><td><b>Answer Relevancy</b></td><td>1.00</td></tr>
<tr><td><b>Answer Correctness</b></td><td>0.92</td></tr>
<tr><td><b>Context Relevance</b></td><td>0.71</td></tr>
</table>

### What the numbers actually mean

Generation is strong across the board — faithfulness and relevancy sit near-ceiling, meaning the model reliably sticks to what it retrieved and directly answers what's asked, even when the retrieved context was imperfect.

**Context Relevance is where the real signal is.** It splits cleanly into two regimes:

- **Broad, single-topic questions** ("What is the case fatality rate?", "What is the incubation period?") → **1.00** context relevance, every time.
- **Narrow, technical-detail questions** ("What genus and family does Nipah belong to?", "What are the two genetic clades?") → **0.33** context relevance — the retriever pulled 3 chunks from the right *document* but only 1 was actually needed, padding the rest of the context with tangential material from elsewhere in that same document.

That's a concrete, actionable finding, not a vague "retrieval could be better": it points directly at `top_k` and chunk-size tuning for technical-detail queries specifically, and it's a finding the evaluation suite surfaced on its own — the generation layer never revealed it, because faithfulness stayed high regardless (the model correctly ignored the irrelevant padding instead of hallucinating from it).

---

## Known issue — found by the eval suite itself

The first full run crashed at question 15/20 with `KeyError: 'relevancy_score'`. Root cause: the relevancy judge occasionally returns malformed JSON, and unlike the faithfulness and context-relevance judges (which already had defensive parsing), the relevancy path had none.

**Fixed** — `_call_judge` now retries once on a parse failure and returns `None` instead of raising, and every metric downstream treats `None` as "unscored" rather than silently coercing it to `0`, so one bad judge call degrades a single metric instead of corrupting an aggregate or killing the run.

The results above are the honest partial numbers from *before* the fix — not backfilled, not estimated. A complete 20/20 run with the fix applied is the next thing to run.

---

## What's next

- [ ] Re-run the full 20-question suite with the parsing fix, including the deliberate negative-test question (no answer exists in the corpus — checks whether the system honestly says so)
- [ ] Tune chunk size / `top_k` specifically against the low-context-relevance question category identified above
- [ ] Judge reliability calibration — compare LLM-judge scores against manual scoring on a held-out sample to quantify how much to trust the automated numbers
- [ ] Regression tracking — re-run the full suite on every pipeline change and diff against the last known-good scores

---

## Setup

```bash
uv add llama-index llama-index-embeddings-huggingface llama-index-vector-stores-faiss faiss-cpu
uv run build_faiss_store.py
uv run run_full_eval.py --out results.json
```

Requires an `OPENROUTER_API_KEY` in `.env` for generation and LLM-as-judge scoring.
