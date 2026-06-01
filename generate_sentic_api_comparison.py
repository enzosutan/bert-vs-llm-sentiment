"""
generate_sentic_api_comparison.py
──────────────────────────────────
Generates notebooks/sentic_api_comparison.ipynb
Run: python generate_sentic_api_comparison.py
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
    "# SenticNet API Comparison\n"
    "\n"
    "> Part of: *BERT vs LLM vs SenticNet: A Multi-Domain Sentiment Comparison*\n"
    "\n"
    "SenticNet takes a different approach from both BERT and LLM methods.\n"
    "It's a knowledge-based commonsense reasoning system — not a statistical model trained\n"
    "on text corpora. This makes it interpretable in a way the other two are not:\n"
    "you can inspect *why* it made a prediction.\n"
    "\n"
    "**APIs used in this notebook:**\n"
    "\n"
    "| API | Key env var | Signal |\n"
    "|-----|-------------|--------|\n"
    "| Ensemble | `SENTIC_ENSEMBLE_KEY` | All signals in one call (primary) |\n"
    "| Polarity | `SENTIC_POLARITY_KEY` | Binary sentiment prediction |\n"
    "| Emotion | `SENTIC_EMOTION_KEY` | Emotion category (JOY, SADNESS, etc.) |\n"
    "| Sarcasm | `SENTIC_SARCASM_KEY` | Sarcasm detection |\n"
    "\n"
    "**API response format** (semicolon-delimited):\n"
    "```\n"
    "POLARITY ; INTENSITY ; EMOTIONS ; INTROSPECTION ; TEMPER ; ATTITUDE ;\n"
    "SENSITIVITY ; PERSONALITY ; ASPECTS ; SARCASM ; DEPRESSION ; TOXICITY ;\n"
    "ENGAGEMENT ; WELL-BEING\n"
    "```\n"
    "\n"
    "---\n"
    "\n"
    "**Latency note:** API calls are ~200-800ms/sample (network). For 2000 samples per domain\n"
    "this will take 30-60 minutes per domain. Start with a smaller subset for exploration."
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
    "\n"
    "sys.path.insert(0, '../src')\n"
    "from data_utils import load_all_domains, SEED, DOMAINS\n"
    "from sentic_utils import run_sentic_inference, summarize_sentic_results, get_emotion_distribution\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "load_dotenv(dotenv_path='../.env')\n"
    "\n"
    "# load all SenticNet API keys\n"
    "KEYS = {\n"
    "    'ensemble':    os.getenv('SENTIC_ENSEMBLE_KEY',    ''),\n"
    "    'polarity':    os.getenv('SENTIC_POLARITY_KEY',    ''),\n"
    "    'emotion':     os.getenv('SENTIC_EMOTION_KEY',     ''),\n"
    "    'sarcasm':     os.getenv('SENTIC_SARCASM_KEY',     ''),\n"
    "    'subjectivity': os.getenv('SENTIC_SUBJECTIVITY_KEY', ''),\n"
    "    'toxicity':    os.getenv('SENTIC_TOXICITY_KEY',    ''),\n"
    "}\n"
    "\n"
    "Path('../results').mkdir(exist_ok=True)\n"
    "Path('../plots').mkdir(exist_ok=True)\n"
    "\n"
    "print('Keys loaded:')\n"
    "for name, key in KEYS.items():\n"
    "    print(f'  {name}: {key[:6]}...' if key else f'  {name}: NOT SET')"
))

cells.append(md("## Load Datasets"))

cells.append(code(
    "datasets = load_all_domains(n_per_domain=2000, dataset_dir='../datasets')\n"
    "\n"
    "# For initial exploration — use a smaller subset to save time.\n"
    "# Increase to 2000 for final results.\n"
    "EXPLORE_N = 200  # change to 2000 for full run\n"
    "print(f'Running Sentic on {EXPLORE_N} samples per domain for initial exploration.')\n"
    "print('Set EXPLORE_N = 2000 for the full dataset.')"
))

cells.append(md(
    "## Quick API Test\n"
    "\n"
    "Before running batch inference, let's verify the API is working and understand\n"
    "the response format with a few example calls."
))

cells.append(code(
    "from sentic_utils import call_sentic_api, clean_for_sentic\n"
    "\n"
    "test_texts = [\n"
    "    'This movie was absolutely fantastic. I loved every minute of it.',\n"
    "    'Terrible waste of time. I want my money back.',\n"
    "    'Yeah right, like this film could ever be considered good. Laughably bad.',  # sarcasm\n"
    "    'The acting was great but the plot was confusing and slow.',  # mixed\n"
    "]\n"
    "\n"
    "print('=== API TEST CALLS ===')\n"
    "for text in test_texts:\n"
    "    result, latency = call_sentic_api(text, KEYS['ensemble'])\n"
    "    print(f'Input:    {text[:60]}...' if len(text) > 60 else f'Input:    {text}')\n"
    "    print(f'Polarity: {result[\"polarity\"]} | Intensity: {result[\"intensity\"]} | '\n"
    "          f'Emotion: {result[\"emotions\"]}')\n"
    "    print(f'Sarcasm:  {result[\"sarcasm\"]} | Is_sarcastic: {result[\"is_sarcastic\"]}')\n"
    "    print(f'Aspects:  {result[\"aspects\"]}')\n"
    "    print(f'Latency:  {latency*1000:.0f}ms')\n"
    "    print()"
))

cells.append(md(
    "### What the API tells us\n"
    "\n"
    "A few things to note from the test calls:\n"
    "\n"
    "- The **intensity** score gives a gradient, not just binary — this is richer than BERT's\n"
    "  softmax confidence or the LLM's single-word output\n"
    "- **Emotion** categories (JOY, SADNESS, ANGER, etc.) are a completely different\n"
    "  representation from the binary positive/negative the other two methods produce\n"
    "- **Aspects** tell you *what* the sentiment is about — this is where SenticNet\n"
    "  is most differentiated from the other methods\n"
    "- **Sarcasm detection** is interesting but imperfect — check how it handles test case 3 above"
))

cells.append(md(
    "## Batch Inference — Ensemble API\n"
    "\n"
    "Using the ensemble key runs all APIs in a single call — efficient and gives\n"
    "the full response including polarity, emotion, sarcasm, aspects etc.\n"
    "\n"
    "⚠️ This is slow. At 0.1s sleep + ~400ms latency per call:\n"
    "- 200 samples ≈ 2 min per domain\n"
    "- 2000 samples ≈ 20 min per domain\n"
    "\n"
    "Results are cached — run once, load from CSV after."
))

cells.append(code(
    "sentic_results = {}\n"
    "summaries = []\n"
    "\n"
    "for domain, df in datasets.items():\n"
    "    cache_path = f'../results/sentic_{domain}.csv'\n"
    "\n"
    "    if Path(cache_path).exists():\n"
    "        print(f'Loading cached {domain} results from {cache_path}')\n"
    "        sentic_df = pd.read_csv(cache_path)\n"
    "        sentic_results[domain] = sentic_df\n"
    "        summary = summarize_sentic_results(sentic_df, df.iloc[:len(sentic_df)]['label'], domain=domain)\n"
    "        summaries.append(summary)\n"
    "        print()\n"
    "        continue\n"
    "\n"
    "    # use subset for exploration\n"
    "    subset = df.head(EXPLORE_N)\n"
    "\n"
    "    print(f'--- {domain.upper()} ({EXPLORE_N} samples) ---')\n"
    "    sentic_df = run_sentic_inference(\n"
    "        subset['text_clean'].tolist(),\n"
    "        key=KEYS['ensemble'],\n"
    "        api_name=f'ensemble/{domain}',\n"
    "        sleep_between=0.1\n"
    "    )\n"
    "    sentic_df['ground_truth'] = subset['label'].values\n"
    "    sentic_df['text'] = subset['text_clean'].values\n"
    "    sentic_df['word_count'] = subset['word_count'].values\n"
    "    sentic_df['domain'] = domain\n"
    "\n"
    "    sentic_df.to_csv(cache_path, index=False)\n"
    "    sentic_results[domain] = sentic_df\n"
    "\n"
    "    summary = summarize_sentic_results(sentic_df, subset['label'], domain=domain)\n"
    "    summaries.append(summary)\n"
    "    print()"
))

cells.append(md("## Summary Table"))

cells.append(code(
    "if summaries:\n"
    "    summary_df = pd.DataFrame(summaries)\n"
    "    summary_df['accuracy'] = summary_df['accuracy'].map('{:.1%}'.format)\n"
    "    summary_df['neutral_rate'] = summary_df['neutral_rate'].map('{:.1%}'.format)\n"
    "    summary_df['avg_latency_ms'] = summary_df['avg_latency_ms'].map('{:.0f} ms'.format)\n"
    "    summary_df['sarcasm_rate'] = summary_df['sarcasm_rate'].map('{:.1%}'.format)\n"
    "    display(summary_df)"
))

cells.append(md(
    "### On neutral predictions\n"
    "\n"
    "SenticNet returns NEUTRAL when it can't determine polarity — this is different from\n"
    "BERT and LLM which always give a binary answer. The 'abstain' rate is actually\n"
    "interesting: it could capture genuine ambiguity. But it also means SenticNet\n"
    "coverage is < 100%, which is a real limitation for production deployment."
))

cells.append(md(
    "## Emotion Distribution by Domain\n"
    "\n"
    "This is unique to SenticNet — the other methods can't give you this."
))

cells.append(code(
    "fig, axes = plt.subplots(1, len(sentic_results), figsize=(14, 5))\n"
    "if len(sentic_results) == 1:\n"
    "    axes = [axes]\n"
    "\n"
    "for ax, (domain, df) in zip(axes, sentic_results.items()):\n"
    "    emo_dist = get_emotion_distribution(df).head(10)\n"
    "    emo_dist.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)\n"
    "    ax.set_title(f'{domain.upper()}\\nEmotion Distribution (top 10)')\n"
    "    ax.set_xlabel('Count')\n"
    "    ax.invert_yaxis()\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('../plots/sentic_emotion_distribution.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved to plots/sentic_emotion_distribution.png')"
))

cells.append(md(
    "### Emotion distribution insights\n"
    "\n"
    "Comparing emotional profiles across domains is a genuinely novel analysis —\n"
    "you can't get this from accuracy numbers alone:\n"
    "\n"
    "- **IMDb** reviews are expressive — expect JOY, SADNESS, ANGER, DISGUST\n"
    "- **Twitter** is more immediate — anger and enthusiasm might dominate\n"
    "- **Amazon** is more functional — CONTENTMENT, ANNOYANCE for good/bad products\n"
    "\n"
    "If the distributions look similar across domains, it suggests SenticNet is\n"
    "applying a domain-agnostic sentiment lens. If they differ, it's capturing\n"
    "genuine domain-level emotional characteristics."
))

cells.append(md(
    "## Sarcasm Analysis\n"
    "\n"
    "How often does SenticNet detect sarcasm, and does it correlate with failures\n"
    "in BERT/LLM? This is the most differentiated capability SenticNet offers."
))

cells.append(code(
    "for domain, df in sentic_results.items():\n"
    "    sarcastic = df[df['is_sarcastic'] == True]\n"
    "    print(f'{domain.upper()}: {len(sarcastic)} sarcastic samples detected ({len(sarcastic)/len(df):.1%})')\n"
    "\n"
    "    if len(sarcastic) > 0:\n"
    "        print('  Sample sarcastic texts:')\n"
    "        for _, row in sarcastic.head(3).iterrows():\n"
    "            print(f'  [{\"POS\" if row[\"ground_truth\"] == 1 else \"NEG\"}] {str(row[\"text\"])[:200]}...')\n"
    "            print(f'  Sarcasm signal: {row[\"sarcasm\"]}')\n"
    "        print()\n"
    "    print()"
))

cells.append(md(
    "## Sarcasm + Accuracy Crosscheck\n"
    "\n"
    "If SenticNet identifies something as sarcastic, does it also predict it correctly?\n"
    "And does that sample cause BERT/LLM to fail?\n"
    "\n"
    "*(Run this cell only after bert_baseline.ipynb and llm_zero_shot.ipynb have been run)*"
))

cells.append(code(
    "for domain in DOMAINS:\n"
    "    if domain not in sentic_results:\n"
    "        continue\n"
    "\n"
    "    sentic_df = sentic_results[domain]\n"
    "    sarcastic_mask = sentic_df['is_sarcastic'].fillna(False)\n"
    "    n_sarcastic = sarcastic_mask.sum()\n"
    "\n"
    "    if n_sarcastic == 0:\n"
    "        print(f'{domain.upper()}: no sarcastic samples detected, skipping.')\n"
    "        continue\n"
    "\n"
    "    # load BERT results for comparison\n"
    "    bert_path = f'../results/bert_{domain}.csv'\n"
    "    if Path(bert_path).exists():\n"
    "        bert_df = pd.read_csv(bert_path).head(len(sentic_df))\n"
    "        sarcastic_idx = sentic_df[sarcastic_mask].index\n"
    "        bert_sarcastic_acc = (bert_df.loc[sarcastic_idx, 'bert_pred'] ==\n"
    "                              sentic_df.loc[sarcastic_idx, 'ground_truth']).mean()\n"
    "        print(f'{domain.upper()}: BERT accuracy on sarcastic samples: {bert_sarcastic_acc:.1%} '\n"
    "              f'(vs {bert_df[\"correct\"].mean():.1%} overall)')\n"
    "    else:\n"
    "        print(f'{domain.upper()}: BERT results not found for comparison')\n"
    "\n"
    "    print()"
))

cells.append(md(
    "## Aspect Extraction — What is Sentiment About?\n"
    "\n"
    "BERT and LLM classify polarity. SenticNet also tells you *what* aspect carries the sentiment.\n"
    "This is the key interpretability advantage."
))

cells.append(code(
    "for domain, df in sentic_results.items():\n"
    "    # look at cases where aspects were actually detected\n"
    "    has_aspects = df[~df['aspects'].str.contains('No aspects', na=True, case=False)]\n"
    "    print(f'{domain.upper()}: {len(has_aspects)}/{len(df)} samples had aspects detected')\n"
    "\n"
    "    if len(has_aspects) > 0:\n"
    "        sample_aspects = has_aspects.sample(min(5, len(has_aspects)), random_state=42)\n"
    "        for _, row in sample_aspects.iterrows():\n"
    "            print(f'  [{\"POS\" if row[\"ground_truth\"] == 1 else \"NEG\"}] aspects: {row[\"aspects\"]}')\n"
    "        print()\n"
    "    print()"
))

cells.append(md(
    "## Latency Comparison\n"
    "\n"
    "SenticNet is slow — it's a knowledge-graph lookup + API call. Let's quantify."
))

cells.append(code(
    "latency_summary = []\n"
    "for domain, df in sentic_results.items():\n"
    "    latency_summary.append({\n"
    "        'domain': domain,\n"
    "        'method': 'SenticNet',\n"
    "        'avg_latency_ms': df['sentic_latency_s'].mean() * 1000,\n"
    "        'p95_latency_ms': df['sentic_latency_s'].quantile(0.95) * 1000,\n"
    "    })\n"
    "\n"
    "latency_df = pd.DataFrame(latency_summary)\n"
    "print('SenticNet latency summary:')\n"
    "display(latency_df)\n"
    "\n"
    "print()\n"
    "print('For comparison:')\n"
    "print('  BERT (CPU):  ~10-50ms/sample')\n"
    "print('  LLM (API):   ~300-800ms/sample')\n"
    "print('  SenticNet:   see above')"
))

cells.append(md(
    "## Expected Characteristics of SenticNet\n"
    "\n"
    "Based on theoretical and pilot explorations, here are the primary differentiators of SenticNet:\n"
    "\n"
    "1. **Symbolic Interpretability:** Unlike statistical black-box models, SenticNet returns specific aspects, "
    "emotion categories, and personality traits, acting as explicit explanations for its decisions.\n"
    "\n"
    "2. **Neutral Abstention:** SenticNet returns NEUTRAL when polarity is ambiguous, acting as a natural "
    "coverage filter rather than forcing a wrong binary label.\n"
    "\n"
    "3. **Explicit Sarcasm Flagging:** The API possesses a dedicated sarcasm parser, which could be extremely "
    "valuable for correcting traditional statistical classifiers on short noisy text like tweets.\n"
    "\n"
    "4. **High Latency Profile:** Since it relies on external Knowledge Graph queries + API requests, it is "
    "expected to be significantly slower than a local BERT model.\n"
    "\n"
    "---\n"
    "\n"
    "### 📝 SenticNet Empirical Evaluation Worksheet\n"
    "**Double-click to edit this cell and document actual findings from your run:**\n"
    "\n"
    "- *Did the API successfully execute, or did you encounter rate-limiting / security block warnings?*\n"
    "- *Compare the average latency of SenticNet with BERT and the LLM from your actual runs:*\n"
    "- *Analyze 1-2 sarcasm-flagged samples. Did the sarcasm detector correctly identify the sarcastic irony?*"
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

out = Path("notebooks/sentic_api_comparison.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out} ({len(cells)} cells)")
