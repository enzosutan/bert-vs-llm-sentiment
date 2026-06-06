# Multi-Domain Sentiment Analysis Comparison

> An empirical comparison of three fundamentally different sentiment analysis approaches — statistical pretrained (BERT), LLM zero-shot (GPT-4o-mini), and commonsense symbolic AI (SenticNet) — across three text domains: IMDb movie reviews, Twitter, and Amazon product reviews.
>
> Student research investigation by Enzo Sutan.

---

## 📊 Empirical Results

All results below are from completed inference runs on **2,000 samples per domain** (BERT and LLM) and **200 samples per domain** (SenticNet), seed=42, balanced classes.

### Accuracy by Method and Domain

| Domain | BERT (DistilBERT) | LLM (GPT-4o-mini) | SenticNet | LLM Advantage |
|--------|:-----------------:|:-----------------:|:---------:|:-------------:|
| **IMDb** | 89.2% | 93.4% | 64.5% | +4.2 pp |
| **Twitter** | 79.1% | 91.0% | 65.1% | **+11.8 pp** |
| **Amazon** | 88.8% | 96.2% | 71.4% | +7.4 pp |

### Speed, Cost, and Latency

| Method | Avg Latency | Cost / 1k samples | Generalization Gap |
|--------|:-----------:|:-----------------:|:-----------------:|
| **BERT** | 27–108 ms | ~$0.00 | 10.1 pp |
| **LLM** | 795–1,109 ms | $0.010–$0.035 | 5.2 pp |
| **SenticNet** | 1,626–2,618 ms | API key (personal) | 6.6 pp |

> **Total LLM cost for all 6,000 samples (3 domains × 2,000): $0.1334**

---

## 🔬 Key Findings

1. **GPT-4o-mini outperforms DistilBERT on every domain without any fine-tuning.** The advantage is largest on Twitter (+11.8 pp) — the domain where BERT struggles most — confirming that LLM contextual reasoning compensates for BERT's register brittleness.

2. **The LLM generalizes more robustly across domains** (5.2 pp spread) than BERT (10.1 pp spread). BERT's performance is strongly tied to text register proximity to its SST-2 training distribution.

3. **Amazon was easier than predicted for both statistical methods.** I expected aspect-level polarity mixing to degrade accuracy, but Amazon (BERT 88.8%, LLM 96.2%) nearly matched IMDb — disconfirming this hypothesis. Amazon product reviews express dominant polarity through locally accessible surface phrases.

4. **BERT's calibration is worst on its hardest domain.** Twitter shows the narrowest calibration gap (0.042) vs IMDb (0.078), meaning BERT is most overconfident precisely where it is most error-prone. Raw BERT confidence cannot be used as a routing signal on Twitter data without temperature scaling.

5. **SenticNet accuracy (64.5–71.4%) is substantially lower** than BERT or LLM, and it does not exhibit meaningful neutral abstention (0–2.5% neutral rate). Its value lies in the qualitative signals it uniquely provides: emotion labels (JOY, ECSTASY, GRIEF), aspect extraction, and sarcasm detection.

6. **Three-way disagreements (all methods differ) are structurally impossible** in binary classification — with only two output classes, all three methods cannot simultaneously disagree. This is a structural constraint, not a finding.

---

## Methods

### BERT
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Pretrained on SST-2 (movie reviews) — no additional training
- CPU inference: 27–108 ms/sample depending on text length
- Truncates at 512 tokens (~350 words)

### LLM (Zero-Shot)
- Model: `gpt-4o-mini`
- Temperature=0, max_tokens=5, zero-shot binary classification
- Domain-aware prompt framing: "this tweet" / "this movie review" / "this product review"
- Input truncated to 200 words (cost control — real accuracy tradeoff)
- Actual costs: $0.071 (IMDb) | $0.021 (Twitter) | $0.042 (Amazon)

### SenticNet
- API: [sentic.net/api](https://sentic.net/api) (ensemble key, 14-field response)
- Returns: polarity, intensity, emotions, aspects, sarcasm, personality, toxicity, engagement, well-being
- 200 samples/domain (latency makes full-dataset runs impractical as primary classifier)
- **Citation:** Cambria et al. *SenticNet 8: Fusing Emotion AI and Commonsense AI for Interpretable, Trustworthy, and Explainable Affective Computing.* HCII 2024. [sentic.net/senticnet-8.pdf](https://sentic.net/senticnet-8.pdf)

---

## Datasets

| Domain | Source | Size | Avg Length | Register |
|--------|--------|:----:|:----------:|----------|
| **IMDb** | HuggingFace `imdb` | 2,000 | 233 words | Expressive, long-form film reviews |
| **Twitter** | HuggingFace `tweet_eval/sentiment` | 2,000 | 20 words | Informal, noisy, context-dependent |
| **Amazon** | HuggingFace `amazon_polarity` | 2,000 | 79 words | Functional, product-focused reviews |

All datasets: balanced (50% pos / 50% neg), seed=42, binary labels, neutral class dropped from Twitter.

---

## Repository Structure

```
multi-domain-sentiment-comparison/
│
├── notebooks/
│   ├── bert_baseline.ipynb          ← BERT on all 3 domains + failure analysis
│   ├── llm_zero_shot.ipynb          ← GPT-4o-mini + cost/latency breakdown
│   ├── sentic_api_comparison.ipynb  ← SenticNet emotion, sarcasm, aspect analysis
│   └── cross_domain_analysis.ipynb  ← Synthesis: tradeoffs, findings, recommendations
│
├── src/
│   ├── data_utils.py     ← Dataset loading + preprocessing
│   ├── bert_utils.py     ← BERT inference helpers
│   ├── llm_utils.py      ← LLM inference + cost estimation
│   └── sentic_utils.py   ← SenticNet API wrapper (14-field parser)
│
├── results/              ← CSV predictions per method per domain (12 files)
│   ├── bert_{imdb,twitter,amazon}.csv
│   ├── llm_{imdb,twitter,amazon}.csv
│   ├── sentic_{imdb,twitter,amazon}.csv
│   └── {bert,llm,sentic}_summary.csv
│
├── datasets/             ← Cached sample CSVs (generated on first run)
├── plots/                ← Saved figures (generated by notebooks)
│
├── run_experiments.py    ← Runs all inference stages end-to-end (cached)
├── analyze_results.py    ← Computes cross-domain metrics from result CSVs
│
├── notebook.ipynb        ← Original Phase 1 pilot: BERT vs LLM, 500 IMDb samples
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/enzosutan/multi-domain-sentiment-comparison.git
cd multi-domain-sentiment-comparison
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY and SenticNet keys

# 3. Run all experiments end-to-end (results cached to results/)
python run_experiments.py

# 4. Or open notebooks individually in Jupyter
jupyter notebook
```

**Notebook run order:**
1. `notebooks/bert_baseline.ipynb` — fast, no API calls needed
2. `notebooks/llm_zero_shot.ipynb` — OpenAI API (~$0.10–0.15 total for all domains)
3. `notebooks/sentic_api_comparison.ipynb` — SenticNet API (slow, use cached results)
4. `notebooks/cross_domain_analysis.ipynb` — synthesis, loads all results, no new API calls

Results are cached to `results/` — once run, notebooks reload from CSV without re-calling APIs.

---

## Recommended Architecture (from findings)

Based on the empirical results, I propose a tiered deployment strategy:

```
Input text
    │
    ▼
[BERT classifier] ─── high confidence ──→ Output (fast, free)
    │
    └── low confidence / informal text
            │
            ▼
      [LLM (gpt-4o-mini)] ──────────────→ Output (slower, ~$0.02/1k)
            │
            └── interpretability needed
                        │
                        ▼
                  [SenticNet] ──────────→ Emotion + aspect output (offline)
```

---

## Known Limitations

- Sample sizes (2k/domain for BERT+LLM, 200/domain for SenticNet) are directional — not statistically robust
- LLM input truncated to 200 words — sentiment in long-tail of reviews is missed
- DistilBERT fine-tuned on SST-2 (movie reviews) — IMDb results near in-distribution
- SenticNet evaluated on 200 samples/domain only — sarcasm analysis has n=9–13 per domain (too small for reliable cross-method comparison)
- No statistical significance testing (McNemar's test is the recommended next step)

---

## References

- Cambria, E. et al. (2024). *SenticNet 8: Fusing Emotion AI and Commonsense AI for Interpretable, Trustworthy, and Explainable Affective Computing.* HCII 2024. [sentic.net/senticnet-8.pdf](https://sentic.net/senticnet-8.pdf)
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS 2019 EMC² Workshop.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in Twitter.* SemEval 2017.
- McAuley, J., Targett, C., Shi, Q., & van den Hengel, A. (2015). *Image-Based Recommendations on Styles and Substitutes.* SIGIR 2015. *(Amazon Polarity dataset)*

---

*Student research investigation. Results are empirical and directional. Not a production system.*
