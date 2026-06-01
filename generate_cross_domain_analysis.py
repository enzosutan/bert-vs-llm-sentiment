"""
generate_cross_domain_analysis.py
──────────────────────────────────
Generates notebooks/cross_domain_analysis.ipynb
Run: python generate_cross_domain_analysis.py

This is the synthesis notebook — loads all results and answers the core research questions.
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
    "# Cross-Domain Analysis\n"
    "\n"
    "> *BERT vs LLM vs SenticNet: A Multi-Domain Sentiment Comparison*\n"
    "\n"
    "This notebook synthesizes results from all three method notebooks.\n"
    "Run this **after** running:\n"
    "1. `bert_baseline.ipynb`\n"
    "2. `llm_zero_shot.ipynb`\n"
    "3. `sentic_api_comparison.ipynb`\n"
    "\n"
    "The goal here is **analysis, not more metrics.**\n"
    "We already have accuracy numbers — this notebook is about understanding *why*.\n"
    "\n"
    "---\n"
    "\n"
    "**Core research questions:**\n"
    "1. Which method generalizes best across domains?\n"
    "2. Does the accuracy gap between methods hold across domains?\n"
    "3. Does SenticNet handle sarcasm better than BERT/LLM?\n"
    "4. Is the LLM cost worth it vs. BERT and SenticNet?\n"
    "5. What are the actual failure patterns — and do they differ by domain?"
))

cells.append(md("## Setup"))

cells.append(code(
    "import sys\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mtick\n"
    "from pathlib import Path\n"
    "\n"
    "sys.path.insert(0, '../src')\n"
    "from data_utils import SEED, DOMAINS\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(SEED)\n"
    "\n"
    "RESULTS_DIR = Path('../results')\n"
    "PLOTS_DIR   = Path('../plots')\n"
    "PLOTS_DIR.mkdir(exist_ok=True)\n"
    "\n"
    "print('Setup complete.')"
))

cells.append(md("## Load All Results"))

cells.append(code(
    "# load all saved CSVs\n"
    "results = {'bert': {}, 'llm': {}, 'sentic': {}}\n"
    "\n"
    "for method in ['bert', 'llm', 'sentic']:\n"
    "    for domain in DOMAINS:\n"
    "        path = RESULTS_DIR / f'{method}_{domain}.csv'\n"
    "        if path.exists():\n"
    "            results[method][domain] = pd.read_csv(path)\n"
    "            print(f'Loaded: {path.name} ({len(results[method][domain])} rows)')\n"
    "        else:\n"
    "            print(f'MISSING: {path.name} — run the corresponding notebook first')\n"
    "\n"
    "print()\n"
    "print('Available results:', {m: list(d.keys()) for m, d in results.items()})"
))

cells.append(md(
    "## Master Comparison Table\n"
    "\n"
    "Accuracy across all methods and domains in one view."
))

cells.append(code(
    "rows = []\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    row = {'domain': domain}\n"
    "\n"
    "    if domain in results['bert']:\n"
    "        bert_df = results['bert'][domain]\n"
    "        row['bert_acc'] = bert_df['correct'].mean() if 'correct' in bert_df.columns else \\\n"
    "                          (bert_df['bert_pred'] == bert_df['ground_truth']).mean()\n"
    "        row['bert_latency_ms'] = bert_df['bert_latency_s'].mean() * 1000\n"
    "    else:\n"
    "        row['bert_acc'] = None; row['bert_latency_ms'] = None\n"
    "\n"
    "    if domain in results['llm']:\n"
    "        llm_df = results['llm'][domain]\n"
    "        valid = llm_df[llm_df['llm_pred'] != -1]\n"
    "        row['llm_acc'] = (valid['llm_pred'] == valid['ground_truth']).mean()\n"
    "        row['llm_latency_ms'] = valid['llm_latency_s'].mean() * 1000\n"
    "        # cost per 1k\n"
    "        total_cost = (llm_df['llm_input_tokens'].sum() / 1e6 * 0.15 +\n"
    "                      llm_df['llm_output_tokens'].sum() / 1e6 * 0.60)\n"
    "        row['llm_cost_per_1k'] = total_cost / len(llm_df) * 1000\n"
    "    else:\n"
    "        row['llm_acc'] = None; row['llm_latency_ms'] = None; row['llm_cost_per_1k'] = None\n"
    "\n"
    "    if domain in results['sentic']:\n"
    "        sentic_df = results['sentic'][domain]\n"
    "        valid = sentic_df[sentic_df['sentic_pred'] != -1]\n"
    "        row['sentic_acc'] = (valid['sentic_pred'] == valid['ground_truth']).mean() if len(valid) > 0 else None\n"
    "        row['sentic_latency_ms'] = sentic_df['sentic_latency_s'].mean() * 1000\n"
    "        row['sentic_neutral_rate'] = (sentic_df['sentic_pred'] == -1).mean()\n"
    "    else:\n"
    "        row['sentic_acc'] = None; row['sentic_latency_ms'] = None; row['sentic_neutral_rate'] = None\n"
    "\n"
    "    rows.append(row)\n"
    "\n"
    "master_df = pd.DataFrame(rows)\n"
    "\n"
    "# display formatted\n"
    "display_df = master_df.copy()\n"
    "for col in ['bert_acc', 'llm_acc', 'sentic_acc', 'sentic_neutral_rate']:\n"
    "    if col in display_df:\n"
    "        display_df[col] = display_df[col].map(lambda x: f'{x:.1%}' if x is not None and not pd.isna(x) else '-')\n"
    "for col in ['bert_latency_ms', 'llm_latency_ms', 'sentic_latency_ms']:\n"
    "    if col in display_df:\n"
    "        display_df[col] = display_df[col].map(lambda x: f'{x:.0f}ms' if x is not None and not pd.isna(x) else '-')\n"
    "if 'llm_cost_per_1k' in display_df:\n"
    "    display_df['llm_cost_per_1k'] = display_df['llm_cost_per_1k'].map(lambda x: f'${x:.3f}' if x is not None and not pd.isna(x) else '-')\n"
    "\n"
    "display(display_df)"
))

cells.append(md(
    "## Accuracy Comparison Plot"
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(9, 5))\n"
    "\n"
    "x = np.arange(len(DOMAINS))\n"
    "width = 0.25\n"
    "colors = {'bert': 'steelblue', 'llm': 'darkorange', 'sentic': 'seagreen'}\n"
    "\n"
    "for i, (method, label) in enumerate([\n"
    "    ('bert',   'BERT (distilbert)'),\n"
    "    ('llm',    'LLM (gpt-4o-mini)'),\n"
    "    ('sentic', 'SenticNet'),\n"
    "]):\n"
    "    accs = [master_df[master_df['domain'] == d][f'{method}_acc'].values[0]\n"
    "            if d in results[method] else 0\n"
    "            for d in DOMAINS]\n"
    "    bars = ax.bar(x + i*width, accs, width, label=label,\n"
    "                  color=colors[method], alpha=0.85, edgecolor='white')\n"
    "\n"
    "ax.set_xlabel('Domain')\n"
    "ax.set_ylabel('Accuracy')\n"
    "ax.set_title('Accuracy by Method and Domain', fontsize=13, fontweight='bold')\n"
    "ax.set_xticks(x + width)\n"
    "ax.set_xticklabels([d.upper() for d in DOMAINS])\n"
    "ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "ax.set_ylim(0, 1.05)\n"
    "ax.legend()\n"
    "ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)\n"
    "ax.text(2.7, 0.51, 'random baseline', fontsize=8, color='gray')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('../plots/cross_domain_accuracy.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved to plots/cross_domain_accuracy.png')"
))

cells.append(md(
    "## Speed vs. Accuracy Tradeoff"
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "\n"
    "markers = {'imdb': 'o', 'twitter': 's', 'amazon': '^'}\n"
    "methods = [\n"
    "    ('bert',   'BERT',      'steelblue'),\n"
    "    ('llm',    'LLM',       'darkorange'),\n"
    "    ('sentic', 'SenticNet', 'seagreen'),\n"
    "]\n"
    "\n"
    "for method, label, color in methods:\n"
    "    for domain in DOMAINS:\n"
    "        if domain not in results[method]:\n"
    "            continue\n"
    "        row = master_df[master_df['domain'] == domain].iloc[0]\n"
    "        acc = row.get(f'{method}_acc')\n"
    "        lat = row.get(f'{method}_latency_ms')\n"
    "        if acc is None or pd.isna(acc) or lat is None or pd.isna(lat):\n"
    "            continue\n"
    "        ax.scatter(lat, acc, marker=markers[domain], color=color, s=100, zorder=5,\n"
    "                   label=f'{label} ({domain})' if domain == 'imdb' else None)\n"
    "        ax.annotate(f'{label[0]}-{domain[:3]}', (lat, acc),\n"
    "                    textcoords='offset points', xytext=(5, 3), fontsize=7)\n"
    "\n"
    "ax.set_xlabel('Avg Latency (ms/sample) — log scale')\n"
    "ax.set_ylabel('Accuracy')\n"
    "ax.set_xscale('log')\n"
    "ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "ax.set_title('Speed vs. Accuracy by Method and Domain', fontsize=12)\n"
    "from matplotlib.lines import Line2D\n"
    "legend_elements = [\n"
    "    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',   markersize=10, label='BERT'),\n"
    "    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',  markersize=10, label='LLM'),\n"
    "    Line2D([0], [0], marker='o', color='w', markerfacecolor='seagreen',    markersize=10, label='SenticNet'),\n"
    "]\n"
    "ax.legend(handles=legend_elements)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../plots/speed_vs_accuracy.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "## Q1: Which method generalizes best across domains?\n"
    "\n"
    "**Generalization Benchmark Goal:** Compare which method's accuracy gap between best-domain and "
    "worst-domain is smallest. A smaller gap = more robust domain generalization.\n"
    "\n"
    "Run the cell below to calculate the actual generalization gaps across all executed domains."
))

cells.append(code(
    "for method in ['bert', 'llm', 'sentic']:\n"
    "    accs = [master_df[master_df['domain'] == d][f'{method}_acc'].values[0]\n"
    "            for d in DOMAINS\n"
    "            if d in results[method] and not pd.isna(master_df[master_df['domain']==d][f'{method}_acc'].values[0])]\n"
    "    if accs:\n"
    "        gap = max(accs) - min(accs)\n"
    "        print(f'{method.upper():10s}: max={max(accs):.1%}  min={min(accs):.1%}  gap={gap:.1%}')"
))

cells.append(md(
    "## Q2: Three-Way Disagreement Analysis\n"
    "\n"
    "Finding samples where ALL THREE methods disagree. These are the genuine hard cases."
))

cells.append(code(
    "three_way = {}\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    if not (domain in results['bert'] and domain in results['llm'] and domain in results['sentic']):\n"
    "        continue\n"
    "\n"
    "    n = min(len(results['bert'][domain]),\n"
    "            len(results['llm'][domain]),\n"
    "            len(results['sentic'][domain]))\n"
    "\n"
    "    b = results['bert'][domain]['bert_pred'].values[:n]\n"
    "    l = results['llm'][domain]['llm_pred'].values[:n]\n"
    "    s = results['sentic'][domain]['sentic_pred'].values[:n]\n"
    "\n"
    "    # exclude cases where any model abstained\n"
    "    valid = (l != -1) & (s != -1)\n"
    "\n"
    "    # three-way disagreement: all different\n"
    "    all_differ = valid & (b != l) & (l != s) & (b != s)\n"
    "\n"
    "    print(f'{domain.upper()}: {all_differ.sum()} three-way disagreements '\n"
    "          f'({all_differ.sum()/valid.sum():.1%} of valid samples)')\n"
    "\n"
    "    three_way[domain] = all_differ"
))

cells.append(md(
    "## Q3: Sarcasm — Does SenticNet Help?\n"
    "\n"
    "Cross-referencing SenticNet's sarcasm flag with BERT/LLM failure rates."
))

cells.append(code(
    "print('Accuracy on sarcasm-flagged samples vs. overall:')\n"
    "print()\n"
    "\n"
    "for domain in DOMAINS:\n"
    "    if domain not in results['sentic']:\n"
    "        continue\n"
    "\n"
    "    sentic_df = results['sentic'][domain]\n"
    "    if 'is_sarcastic' not in sentic_df.columns:\n"
    "        continue\n"
    "\n"
    "    sarcasm_mask = sentic_df['is_sarcastic'].fillna(False)\n"
    "    n = len(sarcasm_mask)\n"
    "    n_sarcastic = sarcasm_mask.sum()\n"
    "\n"
    "    if n_sarcastic == 0:\n"
    "        print(f'{domain.upper()}: no sarcasm detected, skipping')\n"
    "        continue\n"
    "\n"
    "    print(f'{domain.upper()} ({n_sarcastic} sarcastic out of {n}):')\n"
    "\n"
    "    for method, col in [('bert', 'bert_pred'), ('llm', 'llm_pred')]:\n"
    "        if domain not in results[method]:\n"
    "            continue\n"
    "        m_df = results[method][domain].head(n)\n"
    "        gt = sentic_df['ground_truth'].values\n"
    "\n"
    "        overall_acc = (m_df[col].values == gt).mean()\n"
    "        sarc_acc    = (m_df[col].values[sarcasm_mask] == gt[sarcasm_mask]).mean()\n"
    "\n"
    "        print(f'  {method.upper():6s}: overall={overall_acc:.1%} | on sarcastic={sarc_acc:.1%} '\n"
    "              f'(delta={sarc_acc - overall_acc:+.1%})')\n"
    "\n"
    "    print()"
))

cells.append(md(
    "## Q4: Cost/Benefit Summary\n"
    "\n"
    "The practical question: given what we know, how should you choose between methods?"
))

cells.append(code(
    "print('=== PRACTICAL COMPARISON ===')\n"
    "print()\n"
    "print(f'{\"Method\":<15} {\"Cost/1k\":<15} {\"Avg Latency\":<15} {\"Strength\":<40}')\n"
    "print('-' * 90)\n"
    "print(f'{\"BERT\":<15} {\"~$0.00\":<15} {\"10-50ms\":<15} {\"Speed, known failure modes, free\":<40}')\n"
    "print(f'{\"LLM\":<15} {\"~$0.02-0.05\":<15} {\"300-800ms\":<15} {\"Flexibility, reasoning, no training\":<40}')\n"
    "print(f'{\"SenticNet\":<15} {\"API-based\":<15} {\"200-600ms\":<15} {\"Interpretability, sarcasm, emotions\":<40}')\n"
    "print()\n"
    "print('Use case guidance:')\n"
    "print('  High-volume production, clear text:      BERT (+ calibration)')\n"
    "print('  Exploratory / no labels / complex text:  LLM')\n"
    "print('  Need explanation / sarcasm analysis:     SenticNet')\n"
    "print('  Routing strategy:                        BERT confidence → escalate to LLM/Sentic')"
))

cells.append(md(
    "## Synthesis Capstone & Expected Outcomes\n"
    "\n"
    "Synthesizing results across all three paradigms, we have defined the following core expectations "
    "to verify when analyzing our final benchmark:\n"
    "\n"
    "1. **Is Domain Shift the Dominant Effect?**\n"
    "   We expect accuracy to drop more significantly when shifting domains (e.g. from movie reviews "
    "   to tweets) than when switching between architectures on the same domain. The register mismatch is "
    "   a massive headwind for pre-trained systems.\n"
    "\n"
    "2. **Complementary Failure Surfaces:**\n"
    "   - BERT is expected to fail on non-local, structural context.\n"
    "   - The LLM is expected to fail stochastically on subtle contextual nuances or boundaries.\n"
    "   - SenticNet is expected to abstain (NEUTRAL) on borderline cases.\n"
    "\n"
    "3. **Value of Orthogonal Signals:**\n"
    "   SenticNet's specialized aspects, emotions, and sarcasm features represent entirely "
    "   new dimensions of signal that statistical binary classifiers cannot offer.\n"
    "\n"
    "4. **Disagreements as Hard-Case Detectors:**\n"
    "   Cases where BERT, LLM, and SenticNet all disagree are highly likely to represent genuine, "
    "   ambiguous semantic edge cases—making them prime candidates for human-in-the-loop review.\n"
    "\n"
    "---\n"
    "\n"
    "### 📝 Empirical Cross-Domain Synthesis Worksheet\n"
    "**Instructions:** Complete and execute all three individual method notebooks first, then run this "
    "notebook to populate the Master Comparison Table and save the accuracy plots. Double-click here to write your conclusions:\n"
    "\n"
    "- *Which method demonstrated the most robust cross-domain generalization (lowest accuracy variance)?*\n"
    "- *Describe the speed vs. accuracy tradeoff curve plotted above. Is the LLM cost/latency justified in your data?*\n"
    "- *What percentage of your dataset resulted in three-way disagreements? Annotate 1-2 examples of these hard cases:* \n"
    "- *Suggest actionable next steps for a production hybrid system (e.g., routing schemes):*"
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

out = Path("notebooks/cross_domain_analysis.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out} ({len(cells)} cells)")
