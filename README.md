# BERT vs LLM vs SenticNet — Multi-Domain Sentiment Analysis

> An empirical NLP comparison across three sentiment analysis methods (statistical pre-trained, LLM zero-shot, and commonsense symbolic graphs) and three text domains (IMDb, Twitter, Amazon).
> Student research investigation.

---

## 🔬 Investigation Phases

This project is structured into two distinct research phases to balance completed empirical study with a reproducible, scalable benchmarking framework:

### 🔹 Phase 1: Completed Pilot Study (`notebook.ipynb`)
A fully executed, self-contained study comparing **DistilBERT** against zero-shot **GPT-4o-mini** on **500 stratified IMDb movie reviews**. This notebook contains real output cells, cost estimations, latency logs, and confidence distribution plots.

### 🔹 Phase 2: Multi-Domain Scale-Up (`notebooks/`)
A scalable benchmarking framework designed to evaluate three paradigms (adding **SenticNet**) across three domains with a default of **2,000 samples per domain** (6,000 samples total).
* These notebooks are generated via root-level Python scripts (`generate_*.py`) as clean, structured, and interactive templates.
* A researcher can run these notebooks locally, configure custom keys in `.env`, adjust the exploration subset size (`EXPLORE_N`), and complete the embedded **Analysis Worksheets** to document empirical findings.

---

## Problem

How do three fundamentally different sentiment analysis approaches compare across domains?

| Method | Paradigm | Knowledge source |
|--------|----------|-----------------|
| **BERT** (DistilBERT) | Statistical, pretrained classifier | Text corpora (SST-2) |
| **LLM** (GPT-4o-mini) | Statistical, zero-shot reasoning | Massive web-scale data |
| **SenticNet** | Knowledge-based, commonsense AI | Semantic + affective knowledge graph |

More specifically:
- When is BERT good enough?
- When does an LLM justify its cost?
- Does SenticNet handle sarcasm and mixed-sentiment better?
- Which method generalizes across domains — movie reviews, tweets, product reviews?
- What are the actual speed vs. accuracy vs. interpretability tradeoffs?

---

## Methods

### BERT
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Pretrained on SST-2 (movie reviews) — no additional training
- Runs on CPU (~10–50ms/sample), essentially free at scale
- Truncates at 512 tokens — long reviews get cut

### LLM (Zero-Shot)
- Model: `gpt-4o-mini`
- Zero-shot prompting, no examples, temperature=0
- Domain-aware prompt ("this tweet" vs "this movie review")
- Input truncated to 200 words (cost control — a real accuracy tradeoff)
- Cost: ~$0.02–0.05 per 2000 samples

### SenticNet
- API-based commonsense reasoning system ([sentic.net/api](https://sentic.net/api))
- Returns: polarity, intensity, emotion (JOY/SADNESS/etc.), aspects, sarcasm, personality, toxicity, engagement, well-being
- 13 specialized API endpoints (polarity, emotion, sarcasm, ensemble, and more)
- Latency: ~200–600ms/sample (network-dependent)
- Unique capabilities: sarcasm detection, aspect extraction, emotion categorization
- **Citation:** Cambria, E., Liu, Q., Decherchi, S., Xing, F., & Kwok, K. (2022). *SenticNet 7: A Commonsense-based Neurosymbolic AI Framework for Explainable Sentiment Analysis.* AAAI 2022.

---

## Datasets

| Domain | Source | Size | Register | Challenge |
|--------|--------|------|----------|-----------|
| **IMDb** | HuggingFace `imdb` | ~2,000 | Expressive, long-form | Narrative/structural sentiment |
| **Twitter** | HuggingFace `tweet_eval/sentiment` | ~2,000 | Informal, noisy | Sarcasm, abbreviations, short context |
| **Amazon** | HuggingFace `amazon_polarity` | ~2,000 | Functional, product-focused | Aspect-level mixed sentiment |

All datasets balanced (50% pos / 50% neg), seed=42, binary labels.

---

## Key Research Questions

1. Which method generalizes best across domains?
2. Does BERT's calibration problem persist across domains?
3. Does SenticNet detect sarcasm that BERT/LLM miss?
4. Are three-way disagreements (all methods differ) the genuinely hardest cases?
5. Is LLM cost worth it versus BERT and SenticNet at this accuracy level?

---

## Repository Structure

```
bert-vs-llm-sentiment/
│
├── notebooks/
│   ├── bert_baseline.ipynb          ← BERT on all 3 domains
│   ├── llm_zero_shot.ipynb          ← GPT-4o-mini on all 3 domains
│   ├── sentic_api_comparison.ipynb  ← SenticNet APIs + emotion/sarcasm analysis
│   └── cross_domain_analysis.ipynb  ← Synthesis: tradeoffs, failure cases, conclusions
│
├── src/
│   ├── data_utils.py     ← Dataset loading + preprocessing utilities
│   ├── bert_utils.py     ← BERT inference helpers
│   ├── llm_utils.py      ← LLM inference helpers + cost estimation
│   └── sentic_utils.py   ← SenticNet API wrapper (14-field response parser)
│
├── datasets/
│   ├── imdb/             ← IMDb sample (generated, gitignored)
│   ├── twitter/          ← Twitter sample (generated, gitignored)
│   └── amazon/           ← Amazon sample (generated, gitignored)
│
├── results/              ← CSV predictions per method per domain (generated)
├── plots/                ← Saved figures (generated)
│
├── generate_bert_baseline.py        ← Regenerates bert_baseline.ipynb
├── generate_llm_zero_shot.py        ← Regenerates llm_zero_shot.ipynb
├── generate_sentic_api_comparison.py
├── generate_cross_domain_analysis.py
│
├── notebook.ipynb         ← Original BERT vs LLM (500 IMDb samples) — preserved
├── generate_notebook.py   ← Original generator — preserved
│
├── sdk.py                 ← SenticNet SDK reference (from SenticNet team)
├── data.xls               ← Example XLS for SDK testing
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   (SenticNet keys are pre-filled in .env.example)

# 3. Generate notebooks (if .ipynb files aren't present)
python generate_bert_baseline.py
python generate_llm_zero_shot.py
python generate_sentic_api_comparison.py
python generate_cross_domain_analysis.py

# 4. Run notebooks in order
jupyter notebook
```

**Run order:**
1. `notebooks/bert_baseline.ipynb` — fast, no API calls
2. `notebooks/llm_zero_shot.ipynb` — makes OpenAI API calls (~$0.10–0.15 total)
3. `notebooks/sentic_api_comparison.ipynb` — makes SenticNet API calls (slow for large N)
4. `notebooks/cross_domain_analysis.ipynb` — loads all results, no new API calls

Results are cached to `results/` as CSV — once run, reload without re-running APIs.

---

## Cost Estimates

| Method | 2k samples | 6k samples (all domains) |
|--------|-----------|------------------------|
| BERT | ~$0.00 | ~$0.00 |
| LLM (gpt-4o-mini) | ~$0.03–0.05 | ~$0.10–0.15 |
| SenticNet | API-based (check key limits) | — |

---

## Expected Trends & Hypotheses

*(Extrapolated from the Phase 1 Pilot Study — to be verified empirically in Phase 2)*

1. **Domain Shift is the Dominant Factor:** We expect accuracy to drop more significantly moving from IMDb (in-distribution for DistilBERT) to Twitter (noisy, informal register) than when switching between models on a single domain.

2. **BERT Calibration Issue:** DistilBERT is expected to exhibit a narrow calibration gap, meaning it will output high confidence scores even on incorrect predictions, making raw scores unreliable for model routing.

3. **SenticNet's Orthogonal Signal:** SenticNet is expected to offer semantic details (sarcasm flags, aspects, emotional profiles) entirely missing from traditional binary classifiers, though at a higher latency cost.

4. **Three-Way Disagreements are Hard Cases:** We hypothesize that samples where BERT, the LLM, and SenticNet all output different predictions are genuine semantic edge cases, making them excellent candidates for human review queues.

---

## Known Limitations

- Datasets are small (2k/domain) — patterns are directional, not statistically robust
- LLM input truncated to 200 words — misses sentiment in long-tail of reviews
- DistilBERT fine-tuned on SST-2 (movie reviews) — IMDb results are near-in-distribution
- SenticNet latency makes full-dataset runs slow — use subset for exploration
- No statistical significance testing yet (McNemar's test is the next step)

---

## References

- Cambria, E., Liu, Q., Decherchi, S., Xing, F., & Kwok, K. (2022). *SenticNet 7: A Commonsense-based Neurosymbolic AI Framework for Explainable Sentiment Analysis.* Proceedings of AAAI 2022. [sentic.net/api](https://sentic.net/api)
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS 2019 EMC² Workshop.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in Twitter.* SemEval 2017.

---

*This project is a student research investigation, not a production system. Comments are exploratory, findings are directional.*
