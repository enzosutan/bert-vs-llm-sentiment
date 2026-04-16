# BERT vs LLM Sentiment Analysis

> Small-scale, controlled NLP experiment.
> The goal is **understanding model behavior**, not chasing accuracy.

---

## Problem

How do a pretrained BERT classifier and an LLM zero-shot approach compare on binary sentiment classification?

More specifically:
- When is BERT sufficient?
- When does an LLM justify its cost?
- What are the failure modes of each?

---

## Methods

| | BERT | LLM |
|--|------|-----|
| **Model** | `distilbert-base-uncased-finetuned-sst-2-english` | `gpt-4o-mini` |
| **Approach** | Pretrained classifier (no training) | Zero-shot prompting (no training) |
| **Resource class** | Low | High |
| **Cost** | ~$0 (local CPU) | ~$0.04 / 500 samples |
| **Latency** | ~10–50ms/sample | ~300–800ms/sample |

**Dataset:** IMDb reviews — 500 samples (250 positive / 250 negative), fixed seed = 42.

**Prompt design (LLM):** Minimal system prompt + forced output constraint:
> `Answer ONLY with: positive or negative`

---

## Key Findings

1. **BERT is good enough for clear-cut cases.** On unambiguous reviews, both models perform similarly. BERT is ~100× faster and essentially free. No reason to use an LLM for simple, well-formed binary sentiment at scale.

2. **LLM has an edge on structural complexity.** When sentiment builds across a review rather than being expressed in local keywords, BERT fails more often. BERT's failure mode resembles a weighted keyword lookup.

3. **BERT confidence is not well-calibrated.** BERT gives 0.80–0.95 confidence on wrong predictions. The gap between correct/incorrect confidence is smaller than expected. Raw BERT confidence ≠ reliability.

4. **Input truncation has real accuracy consequences.** Truncating LLM input to 200 words for cost control hurt on cases where the review's sentiment resolution was in the second half. This is a real design tradeoff, not just a footnote.

5. **Disagreements are the most valuable signal.** Every case where BERT and LLM disagree is worth inspecting. "Model disagreement" is a cheap, label-free proxy for "this case is ambiguous."

---

## Known Limitations

- Small dataset (500 samples) — patterns are directional, not statistically robust
- No prompt tuning — a better prompt could close or widen the gap
- No BERT fine-tuning — distilBERT used as-is; a task-specific fine-tune is a stronger baseline
- Cost estimates are approximate — token counts vary per review

---

## Reflection

This project reinforced that **failure case analysis beats raw accuracy** as a learning signal. Looking at the cases where BERT and LLM disagree told me more about each model's architecture and training than any accuracy number.

The biggest surprise: LLM failures are harder to predict than BERT failures. BERT has clear failure patterns (non-local sentiment, mixed clauses). LLM failures feel almost random — which could mean genuine model uncertainty, label noise, or both. At 500 samples, you can't tell.

---

## Running the Experiment

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY

# 3. Open the notebook
jupyter notebook notebook.ipynb
```

Estimated total API cost: **< $0.05** for the full 500-sample run.

---

## Structure

```
bert-vs-llm-sentiment/
├── notebook.ipynb          # Full experiment (27 cells, runs end-to-end)
├── generate_notebook.py    # Script that produced notebook.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Data files (`data/sample.csv`, `data/results.csv`) are `.gitignore`d — regenerate by running the notebook.
