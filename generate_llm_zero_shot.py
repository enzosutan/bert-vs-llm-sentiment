"""
generate_llm_zero_shot.py
─────────────────────────
Generates notebooks/llm_zero_shot.ipynb
Run: python generate_llm_zero_shot.py
"""

import json
from pathlib import Path

cells = []
_md = [0]; _code = [0]

def md(source):
    _md[0] += 1
    return {"cell_type": "markdown", "id": f"md-{_md[0]:02d}", "metadata": {}, "source": source}

def code(source):
    _code[0] += 1
    return {"cell_type": "code", "execution_count": None, "id": f"code-{_code[0]:02d}",
            "metadata": {}, "outputs": [], "source": source}


cells.append(md(
    "# LLM Zero-Shot Sentiment — Multi-Domain\n"
    "\n"
    "> Part of: *BERT vs LLM vs SenticNet: A Multi-Domain Sentiment Comparison*\n"
    "\n"
    "This notebook runs GPT-4o-mini zero-shot across the same three domains.\n"
    "\n"
    "**Before running this notebook:**\n"
    "- Make sure `.env` exists with `OPENAI_API_KEY` set\n"
    "- Est. cost: ~2000 × 3 domains × ~150 tokens × $0.15/1M ≈ **$0.10–0.15 total**\n"
    "\n"
    "---\n"
    "\n"
    "No fine-tuning, no examples — pure zero-shot prompting.\n"
    "The prompt is domain-aware (slightly different framing per domain)."
))

cells.append(md("## Setup"))

cells.append(code(
    "import os\n"
    "import sys\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "from dotenv import load_dotenv\n"
    "from openai import OpenAI\n"
    "\n"
    "sys.path.insert(0, '../src')\n"
    "from data_utils import load_all_domains, SEED, DOMAINS\n"
    "from llm_utils import run_llm_inference, summarize_llm_results, estimate_cost, LLM_MODEL\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "load_dotenv(dotenv_path='../.env')\n"
    "\n"
    "OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')\n"
    "if not OPENAI_API_KEY:\n"
    "    raise EnvironmentError('OPENAI_API_KEY not found. Copy .env.example to .env and fill it in.')\n"
    "\n"
    "client = OpenAI(api_key=OPENAI_API_KEY)\n"
    "Path('../results').mkdir(exist_ok=True)\n"
    "Path('../plots').mkdir(exist_ok=True)\n"
    "\n"
    "print(f'Model: {LLM_MODEL}')\n"
    "print('Setup complete.')"
))

cells.append(md("## Load Datasets"))

cells.append(code(
    "datasets = load_all_domains(n_per_domain=2000, dataset_dir='../datasets')\n"
    "for domain, df in datasets.items():\n"
    "    print(f'{domain}: {len(df)} samples | avg words: {df[\"word_count\"].mean():.0f}')"
))

cells.append(md(
    "## Cost Estimate Before Running\n"
    "\n"
    "Let's estimate what this will cost before spending any money.\n"
    "Rough estimate: 2000 samples × ~150 input tokens per sample × 3 domains."
))

cells.append(code(
    "# rough estimate before running\n"
    "est_input_per_sample = 150  # very approximate\n"
    "n_domains = 3\n"
    "n_samples = 2000\n"
    "\n"
    "total_est_tokens = est_input_per_sample * n_samples * n_domains\n"
    "est = estimate_cost(total_est_tokens, n_samples * n_domains)  # ~1 output token each\n"
    "\n"
    "print('=== PRE-RUN COST ESTIMATE ===')\n"
    "print(f'Approx input tokens:  {total_est_tokens:,}')\n"
    "print(f'Estimated total cost: ${est[\"total_cost\"]:.3f}')\n"
    "print()\n"
    "print('This is an estimate — actual will vary based on review length.')\n"
    "print('Twitter tweets are much shorter, so cost will be lower for that domain.')"
))

cells.append(md(
    "## Run Inference — All Domains\n"
    "\n"
    "⚠️ **This cell makes real API calls.** Run it only once per domain if possible.\n"
    "Results are cached to `results/llm_{domain}.csv`.\n"
    "\n"
    "The prompt is slightly domain-adapted: 'this movie review' vs 'this tweet' vs 'this product review'.\n"
    "Temperature=0, max_tokens=5, input truncated to 200 words."
))

cells.append(code(
    "llm_results = {}\n"
    "summaries = []\n"
    "\n"
    "for domain, df in datasets.items():\n"
    "    cache_path = f'../results/llm_{domain}.csv'\n"
    "\n"
    "    # load from cache if already run — avoid re-paying\n"
    "    if Path(cache_path).exists():\n"
    "        print(f'Loading cached {domain} results from {cache_path}')\n"
    "        llm_df = pd.read_csv(cache_path)\n"
    "        llm_results[domain] = llm_df\n"
    "        summary = summarize_llm_results(llm_df, df['label'], domain=domain)\n"
    "        summaries.append(summary)\n"
    "        print()\n"
    "        continue\n"
    "\n"
    "    print(f'--- {domain.upper()} --- (making API calls)')\n"
    "    llm_df = run_llm_inference(\n"
    "        df['text_clean'].tolist(),\n"
    "        client,\n"
    "        domain=domain,\n"
    "        sleep_between=0.05\n"
    "    )\n"
    "    llm_df['correct'] = (llm_df['llm_pred'] == df['label'].values)\n"
    "    llm_df['ground_truth'] = df['label'].values\n"
    "    llm_df['text'] = df['text_clean'].values\n"
    "    llm_df['word_count'] = df['word_count'].values\n"
    "    llm_df['domain'] = domain\n"
    "\n"
    "    llm_df.to_csv(cache_path, index=False)\n"
    "    llm_results[domain] = llm_df\n"
    "\n"
    "    summary = summarize_llm_results(llm_df, df['label'], domain=domain)\n"
    "    summaries.append(summary)\n"
    "    print()"
))

cells.append(md("## Summary Table + Actual Cost"))

cells.append(code(
    "summary_df = pd.DataFrame(summaries)\n"
    "\n"
    "# compute actual total cost across all domains\n"
    "total_input  = sum(llm_results[d]['llm_input_tokens'].sum()  for d in DOMAINS if d in llm_results)\n"
    "total_output = sum(llm_results[d]['llm_output_tokens'].sum() for d in DOMAINS if d in llm_results)\n"
    "actual_cost  = estimate_cost(total_input, total_output)\n"
    "\n"
    "print(f'Total tokens used: {total_input:,} input / {total_output:,} output')\n"
    "print(f'Actual total cost: ${actual_cost[\"total_cost\"]:.4f}')\n"
    "print()\n"
    "\n"
    "summary_df['accuracy'] = summary_df['accuracy'].map('{:.1%}'.format)\n"
    "summary_df['avg_latency_ms'] = summary_df['avg_latency_ms'].map('{:.0f} ms'.format)\n"
    "display(summary_df[['domain', 'accuracy', 'avg_latency_ms', 'total_cost', 'n_samples']])"
))

cells.append(md(
    "## Latency Comparison by Domain"
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    if domain not in llm_results:\n"
    "        continue\n"
    "    latencies = llm_results[domain]['llm_latency_s'] * 1000  # to ms\n"
    "    ax.hist(latencies, bins=30, alpha=0.5, label=domain)\n"
    "\n"
    "ax.set_xlabel('Latency (ms)')\n"
    "ax.set_ylabel('Count')\n"
    "ax.set_title('LLM Latency Distribution by Domain')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig('../plots/llm_latency_by_domain.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "### On latency\n"
    "\n"
    "LLM latency variance is high — network jitter, API server load, response length.\n"
    "Twitter should be faster (shorter input → fewer tokens to process + transmit).\n"
    "IMDb might be the slowest if the prompt + review exceeds the 200-word truncation less often\n"
    "and there's more to encode.\n"
    "\n"
    "In production, this variance matters. You can't guarantee response time the way you can\n"
    "with a local BERT model."
))

cells.append(md("## Failure Case Analysis"))

cells.append(code(
    "def show_llm_failures(llm_df, domain, n=5):\n"
    "    fails = llm_df[~llm_df['correct'] & (llm_df['llm_pred'] != -1)]\n"
    "    sample = fails.sample(min(n, len(fails)), random_state=SEED)\n"
    "    print(f'=== LLM FAILURES [{domain.upper()}] ({len(fails)} total) ===')\n"
    "    for i, (_, row) in enumerate(sample.iterrows()):\n"
    "        gt   = 'POS' if row['ground_truth'] == 1 else 'NEG'\n"
    "        pred = 'POS' if row['llm_pred'] == 1 else 'NEG'\n"
    "        print(f'[{i+1}] Truth:{gt} → LLM:{pred} | raw:\"{row[\"llm_label_raw\"]}\" | words:{row[\"word_count\"]}')\n"
    "        print(f'  {str(row[\"text\"])[:300]}...')\n"
    "        print()\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    if domain in llm_results:\n"
    "        show_llm_failures(llm_results[domain], domain)\n"
    "        print('-' * 70)"
))

cells.append(md(
    "### 🔎 LLM Failure Analysis Guide & Worksheet\n"
    "\n"
    "Based on the Phase 1 Pilot Study (500 IMDb reviews), we have identified several unique behaviors "
    "of LLM zero-shot classification errors. Compare your empirical failures against these guidelines:\n"
    "\n"
    "#### 1. IMDb Failures (Expected: Sarcasm, Truncation, & Over-Reading)\n"
    "- **Sarcasm/Irony:** Zero-shot models sometimes take cynical statements literally.\n"
    "- **200-Word Truncation:** Check if a review's ultimate sentiment was cut off because it exceeded the word threshold.\n"
    "- **Over-reading:** Notice if the LLM hallucinated subtle narrative nuance in a straightforward, angry review.\n"
    "\n"
    "#### 2. Twitter Failures (Expected: Context-dependence)\n"
    "- Short tweets often lack the conversational thread, making sentiment opaque without external context.\n"
    "\n"
    "#### 3. Amazon Failures (Expected: Boundary Ambiguity)\n"
    "- Reviews rated 3-stars (or mixed aspect reviews) are mathematically mapped to POS/NEG but represent borderline cases.\n"
    "\n"
    "---\n"
    "\n"
    "### 📝 Empirical LLM Failure Worksheet\n"
    "**Double-click to edit this cell and document actual failure case patterns you observed:**\n"
    "\n"
    "- *Find and copy a concrete example of a review that failed due to the 200-word truncation limit:* \n"
    "- *Did you find evidence of LLM 'over-reading' or over-rationalizing literal expressions in the Twitter sample?*\n"
    "- *Compare with BERT: Do your LLM failures feel systematically patterned or more stochastic (random)?*"
))

cells.append(md(
    "## Disagreement Analysis — BERT vs LLM\n"
    "\n"
    "Loading BERT results and computing disagreements."
))

cells.append(code(
    "disagreements = {}\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    bert_path = f'../results/bert_{domain}.csv'\n"
    "    if not Path(bert_path).exists():\n"
    "        print(f'BERT results not found for {domain} — run bert_baseline.ipynb first')\n"
    "        continue\n"
    "\n"
    "    bert_df = pd.read_csv(bert_path)\n"
    "    llm_df  = llm_results[domain]\n"
    "\n"
    "    disagree_mask = (bert_df['bert_pred'].values != llm_df['llm_pred'].values)\n"
    "    disagree_mask &= (llm_df['llm_pred'].values != -1)  # exclude LLM errors\n"
    "\n"
    "    n_disagree = disagree_mask.sum()\n"
    "    print(f'{domain.upper()}: {n_disagree}/{len(bert_df)} disagreements ({n_disagree/len(bert_df):.1%})')\n"
    "    disagreements[domain] = disagree_mask"
))

cells.append(md(
    "### Why disagreements matter\n"
    "\n"
    "Model disagreement is a cheap, label-free proxy for 'this sample is ambiguous'.\n"
    "Cases where BERT and LLM disagree are almost always the hardest, most interesting ones\n"
    "— mixed sentiment, sarcasm, unusual structure.\n"
    "\n"
    "If we later compare with SenticNet, we can look at *three-way disagreements*:\n"
    "samples where all three methods give different answers. Those are the genuine edge cases.\n"
    "\n"
    "---\n"
    "\n"
    "*Results saved to `results/llm_{domain}.csv`*"
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/llm_zero_shot.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out} ({len(cells)} cells)")
