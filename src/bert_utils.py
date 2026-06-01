"""
src/bert_utils.py
─────────────────
BERT inference helpers using HuggingFace transformers pipeline.

Model: distilbert-base-uncased-finetuned-sst-2-english
- No training, used as-is (pretrained classifier)
- Runs on CPU (device=-1), no GPU required
- Truncates at 512 tokens (~350 words)
"""

import time
import pandas as pd
from tqdm.notebook import tqdm


def load_bert_pipeline(device: int = -1):
    """
    Load the DistilBERT sentiment pipeline.
    device=-1 means CPU. Pass device=0 for GPU if available.

    First run downloads ~250MB from HuggingFace model hub.
    Subsequent runs use local cache.
    """
    from transformers import pipeline
    print("Loading distilbert pipeline...")
    bert = pipeline(
        'text-classification',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        truncation=True,
        max_length=512,
        device=device,
    )
    print("BERT pipeline ready.")
    return bert


def run_bert_inference(texts: list[str], bert_pipeline, desc: str = "BERT inference") -> pd.DataFrame:
    """
    Run BERT on a list of texts, return DataFrame with predictions + metadata.

    Returns columns:
        bert_pred         — 1 (positive) or 0 (negative)
        bert_label_raw    — raw string from model ('POSITIVE' / 'NEGATIVE')
        bert_confidence   — softmax score (0-1)
        bert_latency_s    — wall-clock seconds per sample
    """
    results = []

    for text in tqdm(texts, desc=desc):
        t0 = time.perf_counter()
        out = bert_pipeline(text)[0]
        t1 = time.perf_counter()
        results.append({
            'bert_pred':       1 if out['label'] == 'POSITIVE' else 0,
            'bert_label_raw':  out['label'],
            'bert_confidence': round(out['score'], 4),
            'bert_latency_s':  round(t1 - t0, 4),
        })

    return pd.DataFrame(results)


def summarize_bert_results(bert_df: pd.DataFrame, ground_truth: pd.Series, domain: str = '') -> dict:
    """
    Compute accuracy + calibration summary.
    Returns dict with key metrics — also prints a report.
    """
    bert_df = bert_df.copy()
    bert_df['correct'] = (bert_df['bert_pred'] == ground_truth.values)

    accuracy = bert_df['correct'].mean()
    avg_latency = bert_df['bert_latency_s'].mean()
    avg_conf = bert_df['bert_confidence'].mean()

    # calibration: does confidence correlate with correctness?
    conf_correct = bert_df[bert_df['correct']]['bert_confidence'].mean()
    conf_wrong   = bert_df[~bert_df['correct']]['bert_confidence'].mean()

    prefix = f"[{domain.upper()}] " if domain else ""
    print(f"{prefix}=== BERT Results ===")
    print(f"  Accuracy:              {accuracy:.1%}")
    print(f"  Avg latency:           {avg_latency * 1000:.1f} ms/sample")
    print(f"  Avg confidence:        {avg_conf:.3f}")
    print(f"  Confidence (correct):  {conf_correct:.3f}")
    print(f"  Confidence (wrong):    {conf_wrong:.3f}  <- calibration gap")

    return {
        'domain': domain,
        'accuracy': accuracy,
        'avg_latency_ms': avg_latency * 1000,
        'avg_confidence': avg_conf,
        'conf_on_correct': conf_correct,
        'conf_on_wrong': conf_wrong,
        'n_samples': len(bert_df),
    }
