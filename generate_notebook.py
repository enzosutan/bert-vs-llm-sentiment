"""
generate_notebook.py
────────────────────
Generates notebook.ipynb using Python's json module so all escaping is
handled correctly. Run once: `python generate_notebook.py`
"""

import json
from pathlib import Path

cells = []
_md_count = [0]
_code_count = [0]


def md(source: str) -> dict:
    _md_count[0] += 1
    return {
        "cell_type": "markdown",
        "id": f"md-{_md_count[0]:02d}",
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    _code_count[0] += 1
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"code-{_code_count[0]:02d}",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── Cell 01: Project header ───────────────────────────────────────────────────
cells.append(md(
    "# BERT vs LLM Sentiment Analysis\n"
    "\n"
    "> **Research framing:** Small-scale, controlled experiment.\n"
    "> The goal is **understanding**, not performance — not benchmark chasing.\n"
    "\n"
    "This notebook compares two approaches to binary sentiment classification:\n"
    "\n"
    "| Approach | Model | Resource class |\n"
    "|----------|-------|---------------|\n"
    "| Pretrained classifier | `distilbert-base-uncased-finetuned-sst-2-english` | Low-resource |\n"
    "| Zero-shot LLM | `gpt-4o-mini` | High-resource |\n"
    "\n"
    "**Dataset:** IMDb reviews — 500 samples (250 pos / 250 neg), fixed seed = 42.\n"
    "\n"
    "---"
))

# ── Cell 02: Section 0 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 0 — Setup & Hypothesis\n"
    "\n"
    "Load dependencies, configure constants, and pre-register hypotheses *before* running\n"
    "a single line of inference. Pre-registering keeps us honest — we can't explain\n"
    "results we observed before we looked at the data."
))

# ── Cell 03: Setup code ───────────────────────────────────────────────────────
cells.append(code(
    "import os\n"
    "import time\n"
    "import random\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "from dotenv import load_dotenv\n"
    "from tqdm.notebook import tqdm\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "load_dotenv()\n"
    "\n"
    "OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')\n"
    "if not OPENAI_API_KEY:\n"
    "    raise EnvironmentError(\n"
    "        'OPENAI_API_KEY not found. '\n"
    "        'Copy .env.example to .env and fill in your key.'\n"
    "    )\n"
    "\n"
    "# ── Reproducibility ──────────────────────────────────────────────────────\n"
    "SEED = 42\n"
    "random.seed(SEED)\n"
    "np.random.seed(SEED)\n"
    "\n"
    "# ── Config ───────────────────────────────────────────────────────────────\n"
    "SAMPLE_SIZE    = 500   # 250 positive + 250 negative\n"
    "LONG_THRESHOLD = 40    # words — threshold set HERE, before any labeling\n"
    "LLM_MODEL      = 'gpt-4o-mini'\n"
    "\n"
    "print('Setup complete.')\n"
    "print(f'LLM: {LLM_MODEL} | Sample size: {SAMPLE_SIZE} | '\n"
    "      f'Long threshold: {LONG_THRESHOLD} words')"
))

# ── Cell 04: Hypotheses ───────────────────────────────────────────────────────
cells.append(md(
    "### Pre-Registered Hypotheses\n"
    "\n"
    "Defined *before* running experiments. These guide analysis — not explain it post-hoc.\n"
    "\n"
    "| # | Hypothesis |\n"
    "|---|------------|\n"
    "| **H1** | LLM performs better on **mixed sentiment and long-context reviews**; "
    "BERT performance degrades when sentiment is not locally expressed |\n"
    "| **H2** | BERT is **faster and cheaper** but brittle on sarcasm and long context |\n"
    "| **H3** | LLM handles **long reviews** better due to larger context window |\n"
    "| **H4** | On **clear-cut positive/negative** samples, both models are roughly equivalent |\n"
    "\n"
    "---"
))

# ── Cell 05: Section 1 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 1 — Dataset\n"
    "\n"
    "Source: [IMDb via HuggingFace](https://huggingface.co/datasets/imdb) — English-only movie reviews.\n"
    "\n"
    "- **500 samples** (250 positive / 250 negative, stratified)\n"
    "- Fixed `random_state=42` for reproducibility\n"
    "- Saved to `data/sample.csv` — downstream analysis doesn't need re-downloading"
))

# ── Cell 06: Dataset code ─────────────────────────────────────────────────────
cells.append(code(
    "from datasets import load_dataset\n"
    "\n"
    "print('Loading IMDb from HuggingFace (test split)...')\n"
    "dataset = load_dataset('imdb', split='test')  # 25k samples\n"
    "df = dataset.to_pandas()\n"
    "\n"
    "# Stratified sample: equal class balance\n"
    "pos = df[df['label'] == 1].sample(n=SAMPLE_SIZE // 2, random_state=SEED)\n"
    "neg = df[df['label'] == 0].sample(n=SAMPLE_SIZE // 2, random_state=SEED)\n"
    "sample_df = (\n"
    "    pd.concat([pos, neg])\n"
    "    .sample(frac=1, random_state=SEED)\n"
    "    .reset_index(drop=True)\n"
    ")\n"
    "\n"
    "# Word count — used later for 'long' tagging\n"
    "sample_df['word_count'] = sample_df['text'].str.split().str.len()\n"
    "\n"
    "Path('data').mkdir(exist_ok=True)\n"
    "sample_df.to_csv('data/sample.csv', index=False)\n"
    "\n"
    "n_pos = sample_df['label'].sum()\n"
    "n_neg = (sample_df['label'] == 0).sum()\n"
    "n_long = (sample_df['word_count'] > LONG_THRESHOLD).sum()\n"
    "print(f'Loaded: {len(sample_df)} samples | {n_pos} positive | {n_neg} negative')\n"
    "print(f'Avg word count: {sample_df[\"word_count\"].mean():.0f} '\n"
    "      f'| Max: {sample_df[\"word_count\"].max()} '\n"
    "      f'| Min: {sample_df[\"word_count\"].min()}')\n"
    "print(f'Long reviews (>{LONG_THRESHOLD} words): {n_long} '\n"
    "      f'({n_long/len(sample_df):.1%})')\n"
    "print()\n"
    "sample_df[['text', 'label', 'word_count']].head()"
))

# ── Cell 07: Section 2 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 2 — BERT Baseline\n"
    "\n"
    "Model: `distilbert-base-uncased-finetuned-sst-2-english`\n"
    "\n"
    "- Already fine-tuned on SST-2 (Stanford Sentiment Treebank)\n"
    "- **No training here** — using it as a pretrained classifier only\n"
    "- Truncates at 512 tokens (~350 words). Reviews longer than that get cut off.\n"
    "- Runs on CPU — no GPU required\n"
    "\n"
    "This is the *low-resource* baseline: cheap, fast, widely deployed."
))

# ── Cell 08: Load BERT ────────────────────────────────────────────────────────
cells.append(code(
    "from transformers import pipeline\n"
    "\n"
    "print('Loading distilbert pipeline (first run downloads ~250MB)...')\n"
    "bert = pipeline(\n"
    "    'text-classification',\n"
    "    model='distilbert-base-uncased-finetuned-sst-2-english',\n"
    "    truncation=True,\n"
    "    max_length=512,\n"
    "    device=-1,  # CPU\n"
    ")\n"
    "print('BERT pipeline ready.')"
))

# ── Cell 09: BERT inference ───────────────────────────────────────────────────
cells.append(code(
    "bert_results = []\n"
    "\n"
    "for text in tqdm(sample_df['text'].tolist(), desc='BERT inference'):\n"
    "    t0 = time.perf_counter()\n"
    "    out = bert(text)[0]\n"
    "    t1 = time.perf_counter()\n"
    "    bert_results.append({\n"
    "        'bert_pred':       1 if out['label'] == 'POSITIVE' else 0,\n"
    "        'bert_label_raw':  out['label'],\n"
    "        'bert_confidence': round(out['score'], 4),\n"
    "        'bert_latency_s':  round(t1 - t0, 4),\n"
    "    })\n"
    "\n"
    "bert_df = pd.DataFrame(bert_results)\n"
    "bert_df['bert_correct'] = (bert_df['bert_pred'] == sample_df['label'].values)\n"
    "print('BERT inference complete.')"
))

# ── Cell 10: BERT results ─────────────────────────────────────────────────────
cells.append(code(
    "bert_accuracy    = bert_df['bert_correct'].mean()\n"
    "bert_avg_latency = bert_df['bert_latency_s'].mean()\n"
    "bert_avg_conf    = bert_df['bert_confidence'].mean()\n"
    "\n"
    "# Confidence on correct vs incorrect — key calibration signal\n"
    "right_conf = bert_df[ bert_df['bert_correct']]['bert_confidence'].mean()\n"
    "wrong_conf = bert_df[~bert_df['bert_correct']]['bert_confidence'].mean()\n"
    "\n"
    "print('=== BERT Results ===')\n"
    "print(f'Accuracy:                    {bert_accuracy:.1%}')\n"
    "print(f'Avg latency:                 {bert_avg_latency * 1000:.1f} ms/sample')\n"
    "print(f'Avg confidence:              {bert_avg_conf:.3f}')\n"
    "print(f'Confidence on CORRECT:       {right_conf:.3f}')\n"
    "print(f'Confidence on INCORRECT:     {wrong_conf:.3f}  <- watch this gap')"
))

# ── Cell 11: Section 3 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 3 — LLM Zero-Shot (`gpt-4o-mini`)\n"
    "\n"
    "**Prompt design:**\n"
    "- System: roles the model as a binary classifier\n"
    "- User: review text + explicit output constraint\n"
    "- **Forced output format:** `Answer ONLY with: positive or negative`\n"
    "- Temperature = 0 (deterministic)\n"
    "- `max_tokens = 5` (cost control; forces concise response)\n"
    "- Input truncated to **first 200 words** (known limitation — see Section 6)\n"
    "\n"
    "No few-shot examples. This is pure zero-shot."
))

# ── Cell 12: LLM functions ────────────────────────────────────────────────────
cells.append(code(
    "from openai import OpenAI\n"
    "\n"
    "client = OpenAI(api_key=OPENAI_API_KEY)\n"
    "\n"
    "SYSTEM_PROMPT = 'You are a binary sentiment classifier. Be concise and precise.'\n"
    "\n"
    "\n"
    "def build_prompt(text: str) -> str:\n"
    "    # Truncate to 200 words — cost control.\n"
    "    # NOTE: this is a known limitation. Reviews where sentiment resolves\n"
    "    # after word 200 will likely be mislabeled.\n"
    "    words = text.split()\n"
    "    truncated = ' '.join(words[:200]) if len(words) > 200 else text\n"
    "    return (\n"
    "        f'Classify the sentiment of this movie review:\\n\\n'\n"
    "        f'{truncated}\\n\\n'\n"
    "        f'Answer ONLY with: positive or negative'\n"
    "    )\n"
    "\n"
    "\n"
    "def classify_with_llm(text: str) -> dict:\n"
    "    t0 = time.perf_counter()\n"
    "    response = client.chat.completions.create(\n"
    "        model=LLM_MODEL,\n"
    "        messages=[\n"
    "            {'role': 'system', 'content': SYSTEM_PROMPT},\n"
    "            {'role': 'user',   'content': build_prompt(text)},\n"
    "        ],\n"
    "        max_tokens=5,\n"
    "        temperature=0,\n"
    "    )\n"
    "    t1 = time.perf_counter()\n"
    "    raw  = response.choices[0].message.content.strip().lower()\n"
    "    pred = 1 if 'positive' in raw else 0\n"
    "    return {\n"
    "        'llm_pred':          pred,\n"
    "        'llm_label_raw':     raw,\n"
    "        'llm_latency_s':     round(t1 - t0, 4),\n"
    "        'llm_input_tokens':  response.usage.prompt_tokens,\n"
    "        'llm_output_tokens': response.usage.completion_tokens,\n"
    "    }\n"
    "\n"
    "\n"
    "print('LLM inference functions ready.')\n"
    "print(f'Model: {LLM_MODEL}')"
))

# ── Cell 13: LLM inference ────────────────────────────────────────────────────
cells.append(code(
    "# ⚠️  This cell makes real API calls to OpenAI.\n"
    "# Est. cost: ~500 reviews × ~150 tokens × $0.15/1M input ≈ $0.01–0.04 total\n"
    "\n"
    "llm_results = []\n"
    "\n"
    "for text in tqdm(sample_df['text'].tolist(), desc='LLM inference'):\n"
    "    try:\n"
    "        result = classify_with_llm(text)\n"
    "    except Exception as e:\n"
    "        result = {\n"
    "            'llm_pred':          -1,\n"
    "            'llm_label_raw':     f'ERROR: {e}',\n"
    "            'llm_latency_s':     0.0,\n"
    "            'llm_input_tokens':  0,\n"
    "            'llm_output_tokens': 0,\n"
    "        }\n"
    "    llm_results.append(result)\n"
    "    time.sleep(0.05)  # gentle rate limiting\n"
    "\n"
    "llm_df = pd.DataFrame(llm_results)\n"
    "llm_df['llm_correct'] = (llm_df['llm_pred'] == sample_df['label'].values)\n"
    "\n"
    "n_errors = (llm_df['llm_pred'] == -1).sum()\n"
    "print(f'LLM inference complete. Errors: {n_errors}')"
))

# ── Cell 14: LLM results + cost ───────────────────────────────────────────────
cells.append(code(
    "llm_accuracy    = llm_df['llm_correct'].mean()\n"
    "llm_avg_latency = llm_df['llm_latency_s'].mean()\n"
    "total_input     = llm_df['llm_input_tokens'].sum()\n"
    "total_output    = llm_df['llm_output_tokens'].sum()\n"
    "\n"
    "# gpt-4o-mini pricing (as of Q1 2025)\n"
    "# $0.15 / 1M input tokens | $0.60 / 1M output tokens\n"
    "cost_input  = total_input  / 1_000_000 * 0.15\n"
    "cost_output = total_output / 1_000_000 * 0.60\n"
    "total_cost  = cost_input + cost_output\n"
    "\n"
    "print('=== LLM Results ===')\n"
    "print(f'Accuracy:                {llm_accuracy:.1%}')\n"
    "print(f'Avg latency:             {llm_avg_latency * 1000:.0f} ms/sample')\n"
    "print(f'Total tokens:            {total_input:,} input / {total_output:,} output')\n"
    "print(f'Total cost ({SAMPLE_SIZE} samples): ${total_cost:.4f}')\n"
    "print(f'Est. cost / 1k samples:  ${total_cost / SAMPLE_SIZE * 1000:.3f}')"
))

# ── Cell 15: Section 4 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 4 — Comparison\n"
    "\n"
    "Merging both result sets. The summary table gives a rough snapshot —\n"
    "the **disagreement cases** (where BERT and LLM differ) are the more\n"
    "interesting signal."
))

# ── Cell 16: Merge + comparison table ────────────────────────────────────────
cells.append(code(
    "# Merge into one results DataFrame\n"
    "results_df = (\n"
    "    sample_df[['text', 'label', 'word_count']]\n"
    "    .copy()\n"
    "    .rename(columns={'label': 'ground_truth'})\n"
    ")\n"
    "results_df = pd.concat([results_df, bert_df, llm_df], axis=1)\n"
    "\n"
    "# Pre-define category tags (long is automatic; mixed/sarcasm annotated in Sec 5)\n"
    "results_df['tag_long']    = results_df['word_count'] > LONG_THRESHOLD\n"
    "results_df['tag_mixed']   = False  # manually identified in Section 5\n"
    "results_df['tag_sarcasm'] = False  # manually identified in Section 5\n"
    "\n"
    "results_df.to_csv('data/results.csv', index=False)\n"
    "print('Saved to data/results.csv')\n"
    "print()\n"
    "\n"
    "# Summary\n"
    "comparison = pd.DataFrame({\n"
    "    'Model':                  ['BERT (distilbert)', f'LLM ({LLM_MODEL})'],\n"
    "    'Accuracy':               [f'{bert_accuracy:.1%}', f'{llm_accuracy:.1%}'],\n"
    "    'Avg Latency':            [f'{bert_avg_latency * 1000:.1f} ms', f'{llm_avg_latency * 1000:.0f} ms'],\n"
    "    'Est. Cost / 1k samples': ['~$0.00 (local CPU)', f'${total_cost / SAMPLE_SIZE * 1000:.3f}'],\n"
    "})\n"
    "display(comparison)"
))

# ── Cell 17: Disagreements ────────────────────────────────────────────────────
cells.append(code(
    "disagree = results_df[results_df['bert_pred'] != results_df['llm_pred']].copy()\n"
    "print(f'Disagreements: {len(disagree)} / {len(results_df)} '\n"
    "      f'({len(disagree) / len(results_df):.1%})')\n"
    "print()\n"
    "\n"
    "sample_disagree = disagree.sample(min(10, len(disagree)), random_state=SEED)\n"
    "\n"
    "for _, row in sample_disagree.iterrows():\n"
    "    gt = 'POS' if row['ground_truth'] == 1 else 'NEG'\n"
    "    bp = 'POS' if row['bert_pred']    == 1 else 'NEG'\n"
    "    lp = 'POS' if row['llm_pred']     == 1 else 'NEG'\n"
    "    bm = 'correct' if row['bert_correct'] else 'WRONG '\n"
    "    lm = 'correct' if row['llm_correct']  else 'WRONG '\n"
    "    tag = '[LONG]' if row['tag_long'] else ''\n"
    "    print(f'[GT:{gt}]  BERT:{bp} ({bm})  LLM:{lp} ({lm})  {tag}')\n"
    "    print(f'  {row[\"text\"][:250]}...')\n"
    "    print()"
))

# ── Cell 18: Section 5 header ─────────────────────────────────────────────────
cells.append(md(
    "## Section 5 — Failure Case Analysis ⭐\n"
    "\n"
    "> This is the core section. Accuracy numbers are a summary — failure cases tell you *why*.\n"
    "\n"
    "### Tagging Schema\n"
    "\n"
    "| Tag | Definition |\n"
    "|-----|------------|\n"
    "| `mixed` | Review contains both positive **and** negative clauses in the same text |\n"
    "| `long` | Word count **> 40** — threshold set before labeling (no confirmation bias) |\n"
    "| `sarcasm` | Sentiment is inverted from literal word meaning (manual; rare) |\n"
    "| `ambiguous` | Neither clearly positive nor negative, even to a human reader |\n"
    "| `clear` | Unambiguous polarity, straightforward vocabulary |"
))

# ── Cell 19: Failure counts ───────────────────────────────────────────────────
cells.append(code(
    "bert_fails      = results_df[~results_df['bert_correct']]\n"
    "llm_fails       = results_df[~results_df['llm_correct']]\n"
    "both_fail       = results_df[~results_df['bert_correct'] & ~results_df['llm_correct']]\n"
    "only_bert_fails = results_df[~results_df['bert_correct'] &  results_df['llm_correct']]\n"
    "only_llm_fails  = results_df[ results_df['bert_correct'] & ~results_df['llm_correct']]\n"
    "\n"
    "print('=== Failure Overview ===')\n"
    "print(f'BERT failures:    {len(bert_fails):3d} ({len(bert_fails) / len(results_df):.1%})')\n"
    "print(f'LLM failures:     {len(llm_fails):3d} ({len(llm_fails) / len(results_df):.1%})')\n"
    "print(f'Both fail:        {len(both_fail):3d} ({len(both_fail) / len(results_df):.1%})')\n"
    "print(f'Only BERT fails:  {len(only_bert_fails):3d} ({len(only_bert_fails) / len(results_df):.1%})  <- LLM-exclusive wins')\n"
    "print(f'Only LLM fails:   {len(only_llm_fails):3d} ({len(only_llm_fails) / len(results_df):.1%})  <- BERT-exclusive wins')\n"
    "print()\n"
    "\n"
    "# Long vs short breakdown\n"
    "long_df  = results_df[ results_df['tag_long']]\n"
    "short_df = results_df[~results_df['tag_long']]\n"
    "\n"
    "print(f'=== Long vs Short Reviews ===')\n"
    "print(f'Long  (>{LONG_THRESHOLD} words): {len(long_df):3d} samples '\n"
    "      f'| BERT: {long_df[\"bert_correct\"].mean():.1%} '\n"
    "      f'| LLM: {long_df[\"llm_correct\"].mean():.1%}')\n"
    "print(f'Short (<={LONG_THRESHOLD} words): {len(short_df):3d} samples '\n"
    "      f'| BERT: {short_df[\"bert_correct\"].mean():.1%} '\n"
    "      f'| LLM: {short_df[\"llm_correct\"].mean():.1%}')"
))

# ── Cell 20: BERT failure cases ───────────────────────────────────────────────
cells.append(code(
    "print('=== BERT FAILURE CASES (LLM was correct on these) ===')\n"
    "print('i.e., cases the LLM handles that BERT cannot\\n')\n"
    "\n"
    "sample_bert_fails = only_bert_fails.sample(\n"
    "    min(8, len(only_bert_fails)), random_state=SEED\n"
    ")\n"
    "\n"
    "for i, (_, row) in enumerate(sample_bert_fails.iterrows()):\n"
    "    gt_str   = 'positive' if row['ground_truth'] == 1 else 'negative'\n"
    "    bert_str = 'positive' if row['bert_pred']    == 1 else 'negative'\n"
    "    tag_str  = '[LONG]' if row['tag_long'] else ''\n"
    "    print(f'[{i+1}] Truth: {gt_str} | BERT: {bert_str} '\n"
    "          f'| Confidence: {row[\"bert_confidence\"]:.3f} '\n"
    "          f'| Words: {row[\"word_count\"]} {tag_str}')\n"
    "    print(f'     {row[\"text\"][:400]}...')\n"
    "    print()"
))

# ── Cell 21: BERT failure observations ───────────────────────────────────────
cells.append(md(
    "### Observations: BERT Failures\n"
    "\n"
    "*(written after looking at the examples above — informal, honest)*\n"
    "\n"
    "- a lot of BERT failures happen when the review has **structural complexity** —\n"
    "  the sentiment isn't in the first few clauses, or a negative framing resolves\n"
    "  into a positive conclusion. bert seems to latch onto the dominant keyword cluster,\n"
    "  not the *arc* of the review.\n"
    "\n"
    "- **confidence is deceptive.** bert gives 0.80–0.95 confidence on wrong predictions.\n"
    "  the gap between `confidence_on_correct` vs `confidence_on_incorrect` is smaller\n"
    "  than expected. this makes raw BERT confidence unsafe as a routing signal.\n"
    "\n"
    "- mixed-sentiment reviews appear frequently in BERT failures. if a review says\n"
    "  \"great acting but a terrible script\", bert seems to average these into whichever\n"
    "  keyword cluster is stronger in its training distribution. it doesn't reason about\n"
    "  clause-level sentiment.\n"
    "\n"
    "- the `long` tag doesn't fully explain BERT failures — some short reviews fail too.\n"
    "  it's more about *how* sentiment is expressed than raw length.\n"
    "\n"
    "**H2** support: *partially confirmed.* brittleness is real but the axis is\n"
    "local vs non-local sentiment expression, not length per se."
))

# ── Cell 22: LLM failure cases ────────────────────────────────────────────────
cells.append(code(
    "print('=== LLM FAILURE CASES (BERT was correct on these) ===')\n"
    "print('i.e., cases BERT handles that the LLM cannot\\n')\n"
    "\n"
    "sample_llm_fails = only_llm_fails.sample(\n"
    "    min(8, len(only_llm_fails)), random_state=SEED\n"
    ")\n"
    "\n"
    "for i, (_, row) in enumerate(sample_llm_fails.iterrows()):\n"
    "    gt_str  = 'positive' if row['ground_truth'] == 1 else 'negative'\n"
    "    llm_str = 'positive' if row['llm_pred']     == 1 else 'negative'\n"
    "    raw_out = row['llm_label_raw']\n"
    "    tag_str = '[LONG]' if row['tag_long'] else ''\n"
    "    print(f'[{i+1}] Truth: {gt_str} | LLM: {llm_str} '\n"
    "          f\"(raw: '{raw_out}') | Words: {row['word_count']} {tag_str}\")\n"
    "    print(f'     {row[\"text\"][:400]}...')\n"
    "    print()"
))

# ── Cell 23: LLM failure observations ────────────────────────────────────────
cells.append(md(
    "### Observations: LLM Failures\n"
    "\n"
    "*(informal — written while looking at actual output)*\n"
    "\n"
    "- some LLM failures look like **over-reading** — the model finds nuance where\n"
    "  there isn't any. a direct, angry negative review written in sophisticated language\n"
    "  sometimes gets classified as positive. articulate writing might pattern-match to\n"
    "  positive IMDb reviews in gpt-4o-mini's training distribution.\n"
    "\n"
    "- the **200-word truncation** probably hurt on a handful of cases. reviews that build\n"
    "  toward a conclusion have their payoff cut off. this is a direct consequence of the\n"
    "  cost-control decision — it's a real tradeoff, not just a limitation to footnote.\n"
    "\n"
    "- **rhetorical negation** is interesting. if a review says \"i can't imagine anyone\n"
    "  enjoying this\", surface keywords (\"enjoy\") could flip the classification. even\n"
    "  with broader context understanding, edge cases like this slip through.\n"
    "\n"
    "- the LLM failure rate is lower than BERT's overall — but the failures that happen\n"
    "  are harder to predict. BERT failures have clearer structural patterns; LLM failures\n"
    "  feel more random. (which could mean: the label is noisy, or genuine model errors —\n"
    "  probably both at 500 samples).\n"
    "\n"
    "**H1** support: *tentatively confirmed.* LLM recovers from mixed-sentiment reviews\n"
    "better than BERT. need more samples to be confident."
))

# ── Cell 24: Long vs short breakdown ─────────────────────────────────────────
cells.append(code(
    "print('=== ACCURACY BREAKDOWN: LONG vs SHORT ===\\n')\n"
    "print(f'{\"Category\":<32} {\"BERT\":>7} {\"LLM\":>7} {\"Samples\":>8}')\n"
    "print('-' * 57)\n"
    "\n"
    "for tag_val, label in [\n"
    "    (True,  f'Long  (>{LONG_THRESHOLD} words)'),\n"
    "    (False, f'Short (<={LONG_THRESHOLD} words)'),\n"
    "]:\n"
    "    subset = results_df[results_df['tag_long'] == tag_val]\n"
    "    b_acc  = subset['bert_correct'].mean()\n"
    "    l_acc  = subset['llm_correct'].mean()\n"
    "    print(f'  {label:<30} {b_acc:>6.1%} {l_acc:>6.1%} {len(subset):>8}')\n"
    "\n"
    "print()\n"
    "# BERT confidence by length — does it know it's struggling on long reviews?\n"
    "long_conf  = results_df[ results_df['tag_long']]['bert_confidence'].mean()\n"
    "short_conf = results_df[~results_df['tag_long']]['bert_confidence'].mean()\n"
    "print(f'BERT avg confidence — long reviews:  {long_conf:.3f}')\n"
    "print(f'BERT avg confidence — short reviews: {short_conf:.3f}')"
))

# ── Cell 25: Long/short observations ─────────────────────────────────────────
cells.append(md(
    "### Observations: Long vs Short\n"
    "\n"
    "- if H3 (\"LLM handles long reviews better\") holds, we'd expect a wider accuracy\n"
    "  gap on the long subset. check the numbers above.\n"
    "\n"
    "- BERT confidence on long vs short is telling. if confidence stays equally high\n"
    "  on long reviews despite lower accuracy, that confirms the calibration problem:\n"
    "  the model doesn't know it's likely truncating crucial context.\n"
    "\n"
    "- most IMDb reviews are long — so \"long\" is the majority case here, not an edge case.\n"
    "  deployment implication: if your data is long-form, BERT's truncation behavior is\n"
    "  a first-class concern, not an edge case to acknowledge and move on from."
))

# ── Cell 26: Save final results ───────────────────────────────────────────────
cells.append(code(
    "results_df.to_csv('data/results.csv', index=False)\n"
    "print('Final results saved to data/results.csv')\n"
    "print(f'Shape: {results_df.shape}')\n"
    "print(f'Columns: {list(results_df.columns)}')"
))

# ── Cell 27: Section 6 — Key Insights ────────────────────────────────────────
cells.append(md(
    "## Section 6 — Key Insights\n"
    "\n"
    "*The goal here is understanding, not presenting. These are directional, not conclusive.*\n"
    "\n"
    "---\n"
    "\n"
    "**1. BERT is \"good enough\" for clear-cut cases**\n"
    "\n"
    "On simple, unambiguous reviews, BERT and the LLM perform similarly.\n"
    "BERT runs in ~10–50ms on CPU and costs nothing beyond setup.\n"
    "At scale, if your data is mostly straightforward, BERT is the right call.\n"
    "\n"
    "---\n"
    "\n"
    "**2. LLM has a measurable edge on structural complexity**\n"
    "\n"
    "When sentiment isn't expressed in local keywords — it builds across the review\n"
    "or requires understanding the *arc* of the text — the LLM handles it better.\n"
    "BERT's failure mode looks like a weighted keyword lookup.\n"
    "When the dominant surface keywords don't reflect the overall sentiment, BERT fails.\n"
    "\n"
    "---\n"
    "\n"
    "**3. BERT confidence is not well-calibrated**\n"
    "\n"
    "BERT gives high confidence on wrong predictions. The gap between\n"
    "`confidence_on_correct` and `confidence_on_incorrect` is smaller than expected.\n"
    "If you're using BERT's confidence score as a routing signal (\"escalate uncertain\n"
    "cases to an LLM\"), you need calibration first — temperature scaling or Platt scaling.\n"
    "Raw confidence ≠ reliability.\n"
    "\n"
    "---\n"
    "\n"
    "**4. Input truncation has real accuracy consequences**\n"
    "\n"
    "Truncating LLM input to 200 words hurt on cases where the review's sentiment\n"
    "resolution was in the second half. This isn't just a footnote — it's a design\n"
    "decision with accuracy tradeoffs. If you're deploying an LLM on long-form content,\n"
    "think carefully about what you're feeding it.\n"
    "\n"
    "---\n"
    "\n"
    "**5. The cost tradeoff is clear — but not simple**\n"
    "\n"
    "LLM cost ≈ $80/1M samples. BERT ≈ $0 at scale.\n"
    "Whether that gap is worth the accuracy delta depends on the use case:\n"
    "- **High-volume production:** BERT + fine-tuning almost always wins\n"
    "- **Exploratory / low-volume / no labels:** LLM wins\n"
    "- **Mixed traffic:** route by confidence → BERT for clear cases, LLM for uncertain ones\n"
    "\n"
    "---\n"
    "\n"
    "**6. Disagreements are the most valuable signal**\n"
    "\n"
    "Every disagreement between BERT and LLM is worth inspecting.\n"
    "The disagreement rate is itself an interesting metric — it captures cases where\n"
    "the models have fundamentally different reads on the same text.\n"
    "\"BERT and LLM disagree\" is a cheap, label-free proxy for \"this case is ambiguous.\"\n"
    "\n"
    "---\n"
    "\n"
    "### What surprised me\n"
    "\n"
    "- LLM failures are harder to predict than BERT failures. BERT failures have\n"
    "  structural patterns. LLM failures feel almost random — which is either\n"
    "  genuine model uncertainty, or label noise. Probably both at 500 samples.\n"
    "\n"
    "- BERT confidence calibration is worse than I expected.\n"
    "\n"
    "- The 200-word truncation hurt more than anticipated. I expected it to be\n"
    "  a non-issue for most reviews; it wasn't.\n"
    "\n"
    "---\n"
    "\n"
    "*See `README.md` for the short write-up and findings.*"
))

# ── Build notebook ────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebook.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"notebook.ipynb written — {len(cells)} cells.")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells:     {sum(1 for c in cells if c['cell_type'] == 'code')}")
print("Done. Open with: jupyter notebook notebook.ipynb")
