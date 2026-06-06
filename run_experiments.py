"""
run_experiments.py
──────────────────
Runs all three sentiment analysis methods across three domains
and saves results to results/ for notebook analysis.

Usage:
    python run_experiments.py

Stages:
    1. Download + cache datasets (IMDb, Twitter, Amazon)
    2. BERT inference on all 3 domains
    3. LLM (gpt-4o-mini) zero-shot on all 3 domains
    4. SenticNet ensemble API on all 3 domains (200-sample subset)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ── Setup paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))
load_dotenv(dotenv_path=ROOT / '.env')

RESULTS_DIR  = ROOT / 'results'
DATASETS_DIR = ROOT / 'datasets'
PLOTS_DIR    = ROOT / 'plots'
for d in [RESULTS_DIR, DATASETS_DIR, PLOTS_DIR]:
    d.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

from data_utils import load_all_domains, DOMAINS

# ── Stage 1: Load datasets ────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1: Loading datasets")
print("=" * 60)
datasets = load_all_domains(n_per_domain=2000, dataset_dir=str(DATASETS_DIR))
print("\nDatasets loaded:")
for domain, df in datasets.items():
    print(f"  {domain}: {len(df)} samples | avg words: {df['word_count'].mean():.0f}")

# ── Stage 2: BERT inference ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: BERT inference (distilbert-sst-2)")
print("=" * 60)

from bert_utils import load_bert_pipeline, run_bert_inference, summarize_bert_results

bert = load_bert_pipeline(device=-1)
bert_summaries = []

for domain, df in datasets.items():
    cache_path = RESULTS_DIR / f'bert_{domain}.csv'
    if cache_path.exists():
        print(f"\n[{domain.upper()}] Loading cached BERT results...")
        bert_df = pd.read_csv(cache_path)
    else:
        print(f"\n[{domain.upper()}] Running BERT inference ({len(df)} samples)...")
        bert_df = run_bert_inference(df['text_clean'].tolist(), bert, desc=f'BERT [{domain}]')
        bert_df['correct']      = (bert_df['bert_pred'] == df['label'].values)
        bert_df['ground_truth'] = df['label'].values
        bert_df['text']         = df['text_clean'].values
        bert_df['word_count']   = df['word_count'].values
        bert_df['domain']       = domain
        bert_df.to_csv(cache_path, index=False)
        print(f"  Saved to {cache_path}")

    summary = summarize_bert_results(bert_df, pd.Series(bert_df['ground_truth']), domain=domain)
    bert_summaries.append(summary)

print("\nBERT SUMMARY TABLE:")
bert_summary_df = pd.DataFrame(bert_summaries)
print(bert_summary_df[['domain', 'accuracy', 'avg_latency_ms', 'conf_on_correct', 'conf_on_wrong', 'n_samples']].to_string(index=False))
bert_summary_df.to_csv(RESULTS_DIR / 'bert_summary.csv', index=False)

# ── Stage 3: LLM inference ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3: LLM zero-shot inference (gpt-4o-mini)")
print("=" * 60)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env — skipping LLM stage.")
else:
    from openai import OpenAI
    from llm_utils import run_llm_inference, summarize_llm_results, estimate_cost, LLM_MODEL
    client = OpenAI(api_key=OPENAI_API_KEY)
    llm_summaries = []

    for domain, df in datasets.items():
        cache_path = RESULTS_DIR / f'llm_{domain}.csv'
        if cache_path.exists():
            print(f"\n[{domain.upper()}] Loading cached LLM results...")
            llm_df = pd.read_csv(cache_path)
        else:
            print(f"\n[{domain.upper()}] Running LLM inference ({len(df)} samples) — this takes ~{len(df)//50} min...")
            llm_df = run_llm_inference(
                df['text_clean'].tolist(),
                client,
                domain=domain,
                sleep_between=0.05
            )
            llm_df['correct']      = (llm_df['llm_pred'] == df['label'].values)
            llm_df['ground_truth'] = df['label'].values
            llm_df['text']         = df['text_clean'].values
            llm_df['word_count']   = df['word_count'].values
            llm_df['domain']       = domain
            llm_df.to_csv(cache_path, index=False)
            print(f"  Saved to {cache_path}")

        summary = summarize_llm_results(llm_df, pd.Series(llm_df['ground_truth']), domain=domain)
        llm_summaries.append(summary)

    total_cost = sum(s.get('total_cost', 0) for s in llm_summaries)
    print(f"\nLLM SUMMARY TABLE:")
    llm_summary_df = pd.DataFrame(llm_summaries)
    print(llm_summary_df[['domain', 'accuracy', 'avg_latency_ms', 'total_cost', 'n_samples']].to_string(index=False))
    llm_summary_df.to_csv(RESULTS_DIR / 'llm_summary.csv', index=False)
    print(f"\nTotal LLM cost across all domains: ${total_cost:.4f}")

# ── Stage 4: SenticNet inference ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 4: SenticNet ensemble API (200 samples per domain)")
print("=" * 60)

SENTIC_KEY = os.getenv('SENTIC_ENSEMBLE_KEY', '')
if not SENTIC_KEY:
    print("ERROR: SENTIC_ENSEMBLE_KEY not found in .env — skipping SenticNet stage.")
else:
    from sentic_utils import run_sentic_inference, summarize_sentic_results
    EXPLORE_N = 200
    sentic_summaries = []

    for domain, df in datasets.items():
        cache_path = RESULTS_DIR / f'sentic_{domain}.csv'
        if cache_path.exists():
            print(f"\n[{domain.upper()}] Loading cached SenticNet results...")
            sentic_df = pd.read_csv(cache_path)
        else:
            subset = df.head(EXPLORE_N)
            print(f"\n[{domain.upper()}] Running SenticNet on {EXPLORE_N} samples (~{EXPLORE_N*0.6/60:.0f} min)...")
            sentic_df = run_sentic_inference(
                subset['text_clean'].tolist(),
                key=SENTIC_KEY,
                api_name=f'ensemble/{domain}',
                sleep_between=0.15
            )
            sentic_df['ground_truth'] = subset['label'].values
            sentic_df['text']         = subset['text_clean'].values
            sentic_df['word_count']   = subset['word_count'].values
            sentic_df['domain']       = domain
            sentic_df.to_csv(cache_path, index=False)
            print(f"  Saved to {cache_path}")

        gt_series = pd.Series(sentic_df['ground_truth'])
        summary = summarize_sentic_results(sentic_df, gt_series, domain=domain)
        if summary:
            sentic_summaries.append(summary)

    if sentic_summaries:
        print(f"\nSenticNet SUMMARY TABLE:")
        sentic_summary_df = pd.DataFrame(sentic_summaries)
        print(sentic_summary_df[['domain', 'accuracy', 'neutral_rate', 'avg_latency_ms', 'sarcasm_rate', 'n_samples']].to_string(index=False))
        sentic_summary_df.to_csv(RESULTS_DIR / 'sentic_summary.csv', index=False)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}")
print("Files written:")
for f in sorted(RESULTS_DIR.glob('*.csv')):
    print(f"  {f.name}")
