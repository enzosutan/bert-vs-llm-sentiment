"""
src/llm_utils.py
────────────────
LLM zero-shot inference helpers using OpenAI gpt-4o-mini.
Refactored from the original notebook.ipynb for reuse across domains.

Cost note (gpt-4o-mini, Q1 2025 pricing):
  $0.15 / 1M input tokens
  $0.60 / 1M output tokens
  Est. ~$0.02-0.05 per 2000-sample domain run
"""

import os
import time
import pandas as pd
from tqdm.notebook import tqdm

# gpt-4o-mini pricing (update this if rates change)
PRICE_PER_1M_INPUT  = 0.15
PRICE_PER_1M_OUTPUT = 0.60

LLM_MODEL = 'gpt-4o-mini'

SYSTEM_PROMPT = 'You are a binary sentiment classifier. Be concise and precise.'

# Domain-specific prompt prefixes — slight variation helps with register
_DOMAIN_CONTEXT = {
    'imdb':    'this movie review',
    'twitter': 'this tweet',
    'amazon':  'this product review',
    'default': 'this text',
}


def build_prompt(text: str, domain: str = 'default', max_words: int = 200) -> str:
    """
    Build the user prompt for zero-shot classification.
    Truncates to max_words — real cost-control tradeoff.
    (Sentiment in the second half of long reviews will be missed.)
    """
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    context = _DOMAIN_CONTEXT.get(domain, _DOMAIN_CONTEXT['default'])
    return (
        f'Classify the sentiment of {context}:\n\n'
        f'{text}\n\n'
        f'Answer ONLY with: positive or negative'
    )


def classify_with_llm(text: str, client, domain: str = 'default',
                      model: str = LLM_MODEL, max_words: int = 200) -> dict:
    """
    Single-sample LLM inference. Returns dict with prediction + usage metadata.
    """
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': build_prompt(text, domain, max_words)},
        ],
        max_tokens=5,
        temperature=0,
    )
    t1 = time.perf_counter()

    raw = response.choices[0].message.content.strip().lower()
    pred = 1 if 'positive' in raw else 0

    return {
        'llm_pred':          pred,
        'llm_label_raw':     raw,
        'llm_latency_s':     round(t1 - t0, 4),
        'llm_input_tokens':  response.usage.prompt_tokens,
        'llm_output_tokens': response.usage.completion_tokens,
    }


def run_llm_inference(texts: list[str], client, domain: str = 'default',
                      model: str = LLM_MODEL, max_words: int = 200,
                      sleep_between: float = 0.05) -> pd.DataFrame:
    """
    Run LLM on a list of texts. Returns DataFrame with predictions + token counts.
    sleep_between: gentle rate limiting (seconds between requests).
    """
    results = []

    for text in tqdm(texts, desc=f'LLM inference [{domain}]'):
        try:
            result = classify_with_llm(text, client, domain, model, max_words)
        except Exception as e:
            result = {
                'llm_pred':          -1,
                'llm_label_raw':     f'ERROR: {e}',
                'llm_latency_s':     0.0,
                'llm_input_tokens':  0,
                'llm_output_tokens': 0,
            }
        results.append(result)
        time.sleep(sleep_between)

    df = pd.DataFrame(results)
    n_errors = (df['llm_pred'] == -1).sum()
    if n_errors:
        print(f"  WARNING: {n_errors} API errors (marked as llm_pred=-1)")

    return df


def estimate_cost(input_tokens: int, output_tokens: int) -> dict:
    """
    Rough cost estimate. Note: these are approximate — token counts vary.
    """
    cost_input  = input_tokens  / 1_000_000 * PRICE_PER_1M_INPUT
    cost_output = output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT
    total = cost_input + cost_output
    return {
        'input_tokens':  input_tokens,
        'output_tokens': output_tokens,
        'cost_input':    round(cost_input, 5),
        'cost_output':   round(cost_output, 5),
        'total_cost':    round(total, 5),
    }


def summarize_llm_results(llm_df: pd.DataFrame, ground_truth: pd.Series,
                          domain: str = '', n_samples: int = None) -> dict:
    """
    Print + return LLM performance summary including cost breakdown.
    """
    llm_df = llm_df.copy()
    llm_df['correct'] = (llm_df['llm_pred'] == ground_truth.values)
    # exclude error rows from accuracy calculation
    valid = llm_df[llm_df['llm_pred'] != -1]

    accuracy = valid['correct'].mean()
    avg_latency = valid['llm_latency_s'].mean()
    cost = estimate_cost(
        llm_df['llm_input_tokens'].sum(),
        llm_df['llm_output_tokens'].sum()
    )

    n = n_samples or len(llm_df)
    prefix = f"[{domain.upper()}] " if domain else ""
    print(f"{prefix}=== LLM Results ===")
    print(f"  Accuracy:              {accuracy:.1%}")
    print(f"  Avg latency:           {avg_latency * 1000:.0f} ms/sample")
    print(f"  Total cost ({n} samples): ${cost['total_cost']:.4f}")
    print(f"  Cost per 1k samples:   ${cost['total_cost'] / n * 1000:.3f}")

    return {
        'domain': domain,
        'accuracy': accuracy,
        'avg_latency_ms': avg_latency * 1000,
        'n_samples': n,
        **cost,
    }
