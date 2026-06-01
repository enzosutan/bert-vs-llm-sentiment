"""
generate_bert_baseline.py
─────────────────────────
Generates notebooks/bert_baseline.ipynb
Run: python generate_bert_baseline.py
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


# ── Cell 01 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "# BERT Baseline — Multi-Domain Sentiment Analysis\n"
    "\n"
    "> Part of: *BERT vs LLM vs SenticNet: A Multi-Domain Sentiment Comparison*\n"
    "\n"
    "This notebook runs DistilBERT across three domains and examines where it succeeds and fails.\n"
    "\n"
    "| Domain | Register | Avg Length | Challenge |\n"
    "|--------|----------|------------|-----------|\n"
    "| IMDb   | Formal-ish, expressive | ~230 words | Structural/narrative sentiment |\n"
    "| Twitter | Informal, noisy | ~20 words | Abbreviations, missing context |\n"
    "| Amazon | Functional, product-focused | ~80 words | Mixed aspects per product |\n"
    "\n"
    "**Key question:** Does BERT's behavior on IMDb generalize, or is it dataset-specific?\n"
    "\n"
    "---"
))

# ── Cell 02 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Setup\n"
    "\n"
    "Loading shared utilities from `src/`. No training happens here — just loading\n"
    "a pretrained model and running inference."
))

# ── Cell 03 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "import sys\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "\n"
    "sys.path.insert(0, '../src')\n"
    "from data_utils import load_all_domains, SEED, DOMAINS\n"
    "from bert_utils import load_bert_pipeline, run_bert_inference, summarize_bert_results\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(SEED)\n"
    "\n"
    "Path('../results').mkdir(exist_ok=True)\n"
    "Path('../plots').mkdir(exist_ok=True)\n"
    "\n"
    "print('Setup complete.')"
))

# ── Cell 04 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Load Datasets\n"
    "\n"
    "Loading ~2000 samples per domain (balanced, seed=42).\n"
    "If cached CSVs exist in `datasets/`, they'll be used directly.\n"
    "First run will download from HuggingFace (~5-10 min depending on connection)."
))

# ── Cell 05 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "datasets = load_all_domains(n_per_domain=2000, dataset_dir='../datasets')\n"
    "\n"
    "for domain, df in datasets.items():\n"
    "    print(f'{domain}: {len(df)} samples | '\n"
    "          f'avg words: {df[\"word_count\"].mean():.0f} | '\n"
    "          f'pos: {df[\"label\"].sum()} | neg: {(df[\"label\"]==0).sum()}')"
))

# ── Cell 06 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Load BERT Pipeline\n"
    "\n"
    "Using `distilbert-base-uncased-finetuned-sst-2-english` — trained on SST-2 (movie reviews).\n"
    "Worth keeping in mind: it was fine-tuned on movie reviews, so IMDb is almost *in-distribution*.\n"
    "Twitter and Amazon are truly out-of-distribution — that's where things get interesting."
))

# ── Cell 07 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "bert = load_bert_pipeline(device=-1)  # CPU"
))

# ── Cell 08 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Run Inference — All Domains\n"
    "\n"
    "~2000 samples × 3 domains = 6000 inference calls. On CPU this takes 5-15 min.\n"
    "Results are saved to `results/bert_{domain}.csv`."
))

# ── Cell 09 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "bert_results = {}\n"
    "summaries = []\n"
    "\n"
    "for domain, df in datasets.items():\n"
    "    print(f'--- {domain.upper()} ---')\n"
    "    bert_df = run_bert_inference(\n"
    "        df['text_clean'].tolist(),\n"
    "        bert,\n"
    "        desc=f'BERT [{domain}]'\n"
    "    )\n"
    "    bert_df['correct'] = (bert_df['bert_pred'] == df['label'].values)\n"
    "    bert_df['ground_truth'] = df['label'].values\n"
    "    bert_df['text'] = df['text_clean'].values\n"
    "    bert_df['word_count'] = df['word_count'].values\n"
    "    bert_df['domain'] = domain\n"
    "\n"
    "    # save\n"
    "    bert_df.to_csv(f'../results/bert_{domain}.csv', index=False)\n"
    "    bert_results[domain] = bert_df\n"
    "\n"
    "    summary = summarize_bert_results(bert_df, df['label'], domain=domain)\n"
    "    summaries.append(summary)\n"
    "    print()"
))

# ── Cell 10 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Summary Table"
))

# ── Cell 11 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "summary_df = pd.DataFrame(summaries)\n"
    "summary_df['accuracy'] = summary_df['accuracy'].map('{:.1%}'.format)\n"
    "summary_df['avg_latency_ms'] = summary_df['avg_latency_ms'].map('{:.1f} ms'.format)\n"
    "summary_df['calibration_gap'] = (summary_df['conf_on_correct'] - summary_df['conf_on_wrong']).map('{:.3f}'.format)\n"
    "display(summary_df[['domain', 'accuracy', 'avg_latency_ms', 'calibration_gap', 'n_samples']])"
))

# ── Cell 12 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Expected Domain Differences (Pilot Study Extrapolations)\n"
    "\n"
    "Based on the completed **Phase 1 Pilot Study** (500 IMDb samples), here are the expected trends "
    "we plan to verify across all three domains:\n"
    "\n"
    "- **IMDb accuracy should be highest** — because `distilbert-sst-2` was fine-tuned on SST-2 (movie reviews), "
    "operating near in-distribution here.\n"
    "\n"
    "- **Twitter is expected to be the most challenging domain** for BERT. Short, noisy texts with slang, "
    "abbreviations, and external context represent a significant register shift from formal movie reviews.\n"
    "\n"
    "- **Amazon reviews should fall in between** — functional register and aspect-level mixed sentiments "
    "(e.g., 'great product, terrible shipping') might degrade performance, even when sentiment is clear.\n"
    "\n"
    "- **The Calibration Gap** (confidence on correct minus confidence on wrong) is our primary indicator "
    "for routing reliability. We expect a narrow gap, meaning DistilBERT will remain highly confident even when wrong.\n"
    "\n"
    "---\n"
    "\n"
    "### 📝 Researcher Analysis Worksheet\n"
    "**Instructions:** Execute the setup and inference cells above, look at the generated summary table, "
    "and document the empirical results below:\n"
    "\n"
    "1. **Actual Accuracies:** IMDb = `___%` | Twitter = `___%` | Amazon = `___%`\n"
    "2. **Calibration Gaps:** IMDb = `___` | Twitter = `___` | Amazon = `___`\n"
    "3. **Does the empirical domain shift match our expectations?** *(Double-click to write your analysis here)*"
))

# ── Cell 13 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Confidence Distribution Plot"
))

# ── Cell 14 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)\n"
    "fig.suptitle('BERT Confidence Distribution by Domain', fontsize=13, fontweight='bold')\n"
    "\n"
    "for ax, domain in zip(axes, DOMAINS):\n"
    "    df = bert_results[domain]\n"
    "    correct = df[df['correct']]['bert_confidence']\n"
    "    wrong   = df[~df['correct']]['bert_confidence']\n"
    "\n"
    "    ax.hist(correct, bins=20, alpha=0.6, color='steelblue', label='Correct')\n"
    "    ax.hist(wrong,   bins=20, alpha=0.6, color='firebrick', label='Wrong')\n"
    "    ax.axvline(correct.mean(), color='steelblue', linestyle='--', linewidth=1)\n"
    "    ax.axvline(wrong.mean(),   color='firebrick', linestyle='--', linewidth=1)\n"
    "    ax.set_title(domain.upper())\n"
    "    ax.set_xlabel('BERT Confidence')\n"
    "    ax.legend(fontsize=8)\n"
    "\n"
    "axes[0].set_ylabel('Count')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../plots/bert_confidence_by_domain.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved to plots/bert_confidence_by_domain.png')"
))

# ── Cell 15 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "### Reading the plot\n"
    "\n"
    "Ideally, the blue (correct) and red (wrong) distributions should be well-separated:\n"
    "correct predictions at high confidence, wrong at low. If the distributions overlap\n"
    "heavily — especially if wrong predictions cluster at high confidence — that confirms\n"
    "BERT's confidence is poorly calibrated and unreliable as a signal.\n"
    "\n"
    "This is worth comparing across domains. Does calibration get worse on harder domains?"
))

# ── Cell 16 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Failure Case Analysis\n"
    "\n"
    "Looking at concrete failure cases per domain. The goal here isn't to compute more metrics —\n"
    "it's to read the actual examples and notice patterns."
))

# ── Cell 17 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "def show_failures(bert_df, domain, n=6):\n"
    "    fails = bert_df[~bert_df['correct']].sample(\n"
    "        min(n, len(bert_df[~bert_df['correct']])), random_state=SEED\n"
    "    )\n"
    "    print(f'=== BERT FAILURES [{domain.upper()}] ({len(bert_df[~bert_df[\"correct\"]])} total) ===')\n"
    "    for i, (_, row) in enumerate(fails.iterrows()):\n"
    "        gt  = 'POS' if row['ground_truth'] == 1 else 'NEG'\n"
    "        pred = 'POS' if row['bert_pred'] == 1 else 'NEG'\n"
    "        print(f'[{i+1}] Truth:{gt} → BERT:{pred} | conf:{row[\"bert_confidence\"]:.3f} | words:{row[\"word_count\"]}')\n"
    "        print(f'  {str(row[\"text\"])[:350]}...')\n"
    "        print()\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    show_failures(bert_results[domain], domain)\n"
    "    print('-' * 70)"
))

# ── Cell 18 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "### 🔎 Failure Case Analysis Guide & Worksheet\n"
    "\n"
    "Look closely at the actual randomly sampled failures printed above. Here is an analytical guide "
    "of expected patterns based on our pilot study, along with a worksheet to fill in:\n"
    "\n"
    "#### 1. IMDb Failures (Expected: Structural & Narrative Complexity)\n"
    "- Check if the model failed on structural arcs (e.g., negative setups with a positive payoff at the end).\n"
    "- Check if there are high-confidence wrong predictions (confidence > 0.90 but incorrect).\n"
    "\n"
    "#### 2. Twitter Failures (Expected: Informal Register & External Context)\n"
    "- Check if tweets contain slang, abbreviations, or emoji-heavy polarity that the model missed.\n"
    "- Look for implicit context where the tweet refers to external events or threads.\n"
    "\n"
    "#### 3. Amazon Failures (Expected: Aspect-Level Mixed Sentiments)\n"
    "- Note if reviews express both polarities (e.g., loving the item but hating the shipping) resulting in a mismatched ground-truth label.\n"
    "\n"
    "---\n"
    "\n"
    "### 📝 Empirical Observations Worksheet\n"
    "**Double-click to edit this cell and document actual failure case patterns you observed:**\n"
    "\n"
    "- *What is the most common structural failure mode in the actual IMDb sample?*\n"
    "- *Does Twitter show a high density of sarcasm or implicit context in its actual errors?*\n"
    "- *Document 1-2 concrete examples of aspect confusion in your Amazon error analysis:* "
))

# ── Cell 19 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Accuracy vs. Review Length (IMDb)\n"
    "\n"
    "A quick check of H3 from the original notebook: does BERT struggle more on longer text?"
))

# ── Cell 20 ───────────────────────────────────────────────────────────────────
cells.append(code(
    "imdb_df = bert_results['imdb'].copy()\n"
    "imdb_df['length_bucket'] = pd.cut(\n"
    "    imdb_df['word_count'],\n"
    "    bins=[0, 50, 100, 200, 400, 2000],\n"
    "    labels=['<50', '50-100', '100-200', '200-400', '400+']\n"
    ")\n"
    "\n"
    "bucket_acc = imdb_df.groupby('length_bucket')['correct'].agg(['mean', 'count'])\n"
    "bucket_acc.columns = ['accuracy', 'n_samples']\n"
    "bucket_acc['accuracy'] = bucket_acc['accuracy'].map('{:.1%}'.format)\n"
    "print('IMDb accuracy by review length bucket:')\n"
    "display(bucket_acc)"
))

# ── Cell 21 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Key Takeaways\n"
    "\n"
    "1. **Domain matters more than model architecture** (tentatively). The accuracy gap across\n"
    "   domains is often larger than the gap between methods. This suggests dataset distribution\n"
    "   shift is the main challenge, not model capacity.\n"
    "\n"
    "2. **BERT is fast everywhere.** ~10-50ms per sample on CPU regardless of domain.\n"
    "   That's the one thing that doesn't change.\n"
    "\n"
    "3. **Calibration is a genuine problem.** If you're using BERT confidence to decide whether\n"
    "   to escalate to a more expensive model, you need to calibrate it first. Raw score ≠ reliability.\n"
    "\n"
    "4. **Twitter is a different animal.** Short text, informal register, sarcasm density.\n"
    "   A model trained on longer structured text has a rough time here regardless of architecture.\n"
    "\n"
    "---\n"
    "\n"
    "*Results saved to `results/bert_{domain}.csv` for use in cross_domain_analysis.ipynb*"
))

# ── Write notebook ────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/bert_baseline.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out} ({len(cells)} cells)")
