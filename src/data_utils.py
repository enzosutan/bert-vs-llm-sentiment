"""
src/data_utils.py
─────────────────
Shared dataset loading + preprocessing helpers.

Usage from notebooks (add to sys.path first):
    import sys; sys.path.insert(0, '../src')
    from data_utils import load_imdb_sample, load_twitter_sample, load_amazon_sample
"""

import re
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

# shared constants — set these once, use everywhere
SEED = 42
DOMAINS = ['imdb', 'twitter', 'amazon']

# Characters that break the SenticNet API — must be stripped/replaced
_SENTIC_ILLEGAL = ['&', '#', ';', '{', '}']


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def preprocess_text(text: str) -> str:
    """Basic cleaning shared across all methods.
    - Strip HTML line breaks (common in IMDb)
    - Remove leading/trailing whitespace
    Does NOT remove SenticNet-illegal chars — that's done separately in sentic_utils.py
    so other pipelines aren't affected.
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_for_sentic(text: str, max_words: int = 300) -> str:
    """
    Preprocessing specifically for SenticNet API input.
    - Removes illegal characters: & # ; { }
    - Truncates to max_words (API recommends ~1000 words, we're conservative)
    - Strips HTML artifacts
    """
    text = preprocess_text(text)
    for char in _SENTIC_ILLEGAL:
        text = text.replace(char, ':')
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Rough sentence splitter for sentence-level Sentic API calls.
    Not using nltk to keep dependencies minimal — simple period/!/?  split.
    Good enough for exploratory work.
    """
    # split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # drop empties and very short fragments
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def load_imdb_sample(n: int = 2000, seed: int = SEED, save_path: str = None) -> pd.DataFrame:
    """
    Load n samples from the IMDb dataset (HuggingFace), balanced pos/neg.
    Returns DataFrame with: text, label, domain, word_count, text_clean
    """
    from datasets import load_dataset  # lazy import — not everyone needs this

    print(f"Loading IMDb (test split) from HuggingFace...")
    dataset = load_dataset('imdb', split='test')
    df = dataset.to_pandas()

    half = n // 2
    pos = df[df['label'] == 1].sample(n=half, random_state=seed)
    neg = df[df['label'] == 0].sample(n=half, random_state=seed)
    sample = (
        pd.concat([pos, neg])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    sample['domain'] = 'imdb'
    sample['word_count'] = sample['text'].str.split().str.len()
    sample['text_clean'] = sample['text'].apply(preprocess_text)

    print(f"  IMDb: {len(sample)} samples | {sample['label'].sum()} pos | {(sample['label']==0).sum()} neg")
    print(f"  Avg word count: {sample['word_count'].mean():.0f} | Max: {sample['word_count'].max()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

    return sample


def load_twitter_sample(n: int = 2000, seed: int = SEED, save_path: str = None) -> pd.DataFrame:
    """
    Load n samples from tweet_eval/sentiment (HuggingFace).
    Original labels: 0=negative, 1=neutral, 2=positive
    We drop neutral and remap: 0=negative, 1=positive (binary)

    Twitter text is short — average ~20-30 words. Very different from IMDb.
    This domain is useful for testing how models handle short, informal text.
    """
    from datasets import load_dataset

    print(f"Loading tweet_eval/sentiment from HuggingFace...")
    # using the full split to get enough after dropping neutrals
    dataset = load_dataset('tweet_eval', 'sentiment', split='train')
    df = dataset.to_pandas()

    # drop neutral (label == 1)
    df = df[df['label'] != 1].copy()
    # remap: 2 -> 1 (positive), 0 stays 0 (negative)
    df['label'] = df['label'].map({0: 0, 2: 1})

    half = n // 2
    pos = df[df['label'] == 1].sample(n=half, random_state=seed)
    neg = df[df['label'] == 0].sample(n=half, random_state=seed)
    sample = (
        pd.concat([pos, neg])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    sample['domain'] = 'twitter'
    sample['word_count'] = sample['text'].str.split().str.len()
    sample['text_clean'] = sample['text'].apply(preprocess_text)

    print(f"  Twitter: {len(sample)} samples | {sample['label'].sum()} pos | {(sample['label']==0).sum()} neg")
    print(f"  Avg word count: {sample['word_count'].mean():.0f} | Max: {sample['word_count'].max()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

    return sample


def load_amazon_sample(n: int = 2000, seed: int = SEED, save_path: str = None) -> pd.DataFrame:
    """
    Load n samples from amazon_polarity dataset (HuggingFace).
    Labels: 0=negative, 1=positive. Already binary.

    Amazon reviews are medium-length (50-200 words typically).
    Different register than IMDb — more product-focused, often concise.
    Interesting domain because language is more functional than expressive.
    """
    from datasets import load_dataset

    print(f"Loading amazon_polarity (test split) from HuggingFace...")
    # test split has 400k rows — more than enough
    dataset = load_dataset('amazon_polarity', split='test', trust_remote_code=True)
    df = dataset.to_pandas()

    # combine title + content — both carry sentiment signal
    df['text'] = df['title'].fillna('') + '. ' + df['content'].fillna('')

    half = n // 2
    pos = df[df['label'] == 1].sample(n=half, random_state=seed)
    neg = df[df['label'] == 0].sample(n=half, random_state=seed)
    sample = (
        pd.concat([pos, neg])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    sample = sample[['text', 'label']].copy()
    sample['domain'] = 'amazon'
    sample['word_count'] = sample['text'].str.split().str.len()
    sample['text_clean'] = sample['text'].apply(preprocess_text)

    print(f"  Amazon: {len(sample)} samples | {sample['label'].sum()} pos | {(sample['label']==0).sum()} neg")
    print(f"  Avg word count: {sample['word_count'].mean():.0f} | Max: {sample['word_count'].max()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

    return sample


def load_all_domains(n_per_domain: int = 2000, seed: int = SEED,
                     dataset_dir: str = '../datasets') -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper — load all three domains.
    Returns dict: {'imdb': df, 'twitter': df, 'amazon': df}
    Will load from saved CSVs if they exist (avoids re-downloading).
    """
    loaders = {
        'imdb': (load_imdb_sample, f'{dataset_dir}/imdb/imdb_sample.csv'),
        'twitter': (load_twitter_sample, f'{dataset_dir}/twitter/twitter_sample.csv'),
        'amazon': (load_amazon_sample, f'{dataset_dir}/amazon/amazon_sample.csv'),
    }

    data = {}
    for domain, (loader_fn, cache_path) in loaders.items():
        if Path(cache_path).exists():
            print(f"Loading {domain} from cache: {cache_path}")
            data[domain] = pd.read_csv(cache_path)
        else:
            data[domain] = loader_fn(n=n_per_domain, seed=seed, save_path=cache_path)
        print()

    return data
