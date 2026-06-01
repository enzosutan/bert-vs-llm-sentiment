"""
src/sentic_utils.py
───────────────────
SenticNet API integration helpers.

API endpoint format:
    https://sentic.net/api/{LANG}/{KEY}.py?text={TEXT}

Response format (semicolon-delimited string):
    POLARITY ; INTENSITY ; EMOTIONS_SCORE ; INTROSPECTION ; TEMPER ; ATTITUDE ;
    SENSITIVITY ; PERSONALITY ; ASPECTS ; SARCASM ; DEPRESSION ; TOXICITY ;
    ENGAGEMENT ; WELL-BEING

Example: "POSITIVE;45;JOY;12;8;5;9;Openness;product quality;No sarcasm detected;0%;15%;67%;72%"

Discovered by reading sdk.py (the sample SDK from SenticNet).
The ensemble key (JncOmjr3n5ptX) runs ALL APIs in one call — most efficient for comparison.
"""

import re
import time
import requests
import pandas as pd
from tqdm.notebook import tqdm

LANG = 'en'
BASE_URL = 'https://sentic.net/api/{lang}/{key}.py?text={text}'

# User-Agent to match the SDK (avoids some server-side blocks)
_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/111.0.0.0 Safari/537.36'
    )
}

# Characters that cause "Internal Server Error" from the API
_ILLEGAL_CHARS = ['&', '#', ';', '{', '}']

# Column names for the semicolon-delimited response fields
# (derived from sdk.py header definitions)
_RESPONSE_FIELDS = [
    'polarity',       # POSITIVE / NEGATIVE / NEUTRAL
    'intensity',      # numeric, sentiment intensity
    'emotions',       # emotion label (JOY, SADNESS, etc.)
    'introspection',  # numeric score
    'temper',         # numeric score
    'attitude',       # numeric score
    'sensitivity',    # numeric score
    'personality',    # personality trait string
    'aspects',        # aspect phrases extracted
    'sarcasm',        # sarcasm detection result
    'depression',     # percentage string
    'toxicity',       # percentage string
    'engagement',     # percentage string
    'wellbeing',      # percentage string
]

# Fallback when API returns Internal Server Error
_FALLBACK_RESPONSE = (
    'NEUTRAL;0;No emotions detected;0;0;0;0;'
    'No personality trait detected;No aspects discovered;'
    'No sarcasm detected;0%;0%;0%;0%'
)


def clean_for_sentic(text: str, max_words: int = 300) -> str:
    """
    Prepare text for SenticNet API submission.
    - Replace illegal chars with ':'
    - Truncate to max_words (API handles up to ~1000 words but we're conservative)
    - Strip HTML artifacts
    """
    # basic HTML cleanup
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # remove illegal characters
    for c in _ILLEGAL_CHARS:
        text = text.replace(c, ':')

    # truncate
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    return text


def _parse_response(raw: str) -> dict:
    """
    Parse the semicolon-delimited API response string into a dict.
    Also decodes HTML entities that appear in intensity/score fields.
    """
    raw = raw.replace('&#8593;', '↑').replace('&#8595;', '↓')
    values = raw.split(';')

    result = {}
    for i, field in enumerate(_RESPONSE_FIELDS):
        result[field] = values[i].strip() if i < len(values) else None

    # derived binary prediction (1=positive, 0=negative, -1=neutral/unknown)
    pol = result.get('polarity', '').upper()
    if pol == 'POSITIVE':
        result['sentic_pred'] = 1
    elif pol == 'NEGATIVE':
        result['sentic_pred'] = 0
    else:
        result['sentic_pred'] = -1  # NEUTRAL or parse failure

    # try to parse intensity as float
    try:
        result['intensity_val'] = float(result['intensity'])
    except (ValueError, TypeError):
        result['intensity_val'] = 0.0

    # sarcasm: "No sarcasm detected" → False, anything else → True
    # (single-API calls may return None here if fewer fields are present)
    sarcasm_raw = result.get('sarcasm') or ''
    sarcasm_str = sarcasm_raw.lower()
    result['is_sarcastic'] = (
        bool(sarcasm_str) and
        'no sarcasm' not in sarcasm_str and
        'not detected' not in sarcasm_str
    )

    return result


_WAF_WARNING_SHOWN = False


def call_sentic_api(text: str, key: str, lang: str = LANG,
                    timeout: int = 10) -> tuple[dict, float]:
    """
    Make one API call to SenticNet. Returns (parsed_dict, latency_seconds).
    On error, returns (fallback_dict, latency_seconds).
    """
    global _WAF_WARNING_SHOWN
    cleaned_text = clean_for_sentic(text)
    url = BASE_URL.format(lang=lang, key=key, text=cleaned_text)

    t0 = time.perf_counter()
    raw = _FALLBACK_RESPONSE
    api_blocked = False
    
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 465 or 'Access Denied' in resp.text or 'blocked' in resp.text.lower():
            api_blocked = True
            raw = _FALLBACK_RESPONSE
            if not _WAF_WARNING_SHOWN:
                print("\n[WARNING] SenticNet API requests are being blocked by a Web Application Firewall (WAF) / security block!")
                print("Your IP or key has been restricted. Standard 'NEUTRAL' fallback responses will be used.")
                print("To resolve this, try switching networks, checking your API keys, or using a smaller subset.\n")
                _WAF_WARNING_SHOWN = True
        else:
            resp.raise_for_status()
            raw = resp.text.strip()
            if not raw or 'Internal Server Error' in raw:
                raw = _FALLBACK_RESPONSE
    except Exception as e:
        raw = _FALLBACK_RESPONSE
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 465 or 'Access Denied' in e.response.text:
                api_blocked = True
                if not _WAF_WARNING_SHOWN:
                    print("\n[WARNING] SenticNet API requests are being blocked by a security filter (Status 465)!")
                    _WAF_WARNING_SHOWN = True
    t1 = time.perf_counter()

    parsed = _parse_response(raw)
    parsed['raw_response'] = raw
    parsed['api_blocked'] = api_blocked
    return parsed, round(t1 - t0, 4)


def run_sentic_inference(texts: list[str], key: str, api_name: str = 'ensemble',
                         sleep_between: float = 0.1) -> pd.DataFrame:
    """
    Run a SenticNet API across a list of texts.
    Returns DataFrame with all parsed fields + latency.

    sleep_between: seconds between requests (be polite to the API server).
    For the ensemble API, this gives you all fields in one call.
    """
    results = []

    for text in tqdm(texts, desc=f'SenticNet [{api_name}]'):
        parsed, latency = call_sentic_api(text, key)
        parsed['sentic_latency_s'] = latency
        results.append(parsed)
        time.sleep(sleep_between)

    return pd.DataFrame(results)


def summarize_sentic_results(sentic_df: pd.DataFrame, ground_truth: pd.Series,
                             domain: str = '') -> dict:
    """
    Compute accuracy + sarcasm detection summary.
    Note: NEUTRAL predictions (-1) are excluded from accuracy, similar to "abstain".
    """
    df = sentic_df.copy()
    df['ground_truth'] = ground_truth.values

    # only score non-neutral predictions
    valid = df[df['sentic_pred'] != -1]
    neutral_count = (df['sentic_pred'] == -1).sum()

    if len(valid) == 0:
        print("WARNING: all predictions are NEUTRAL — API may have failed.")
        return {}

    accuracy = (valid['sentic_pred'] == valid['ground_truth']).mean()
    avg_latency = df['sentic_latency_s'].mean()
    sarcasm_rate = df['is_sarcastic'].mean()

    prefix = f"[{domain.upper()}] " if domain else ""
    print(f"{prefix}=== SenticNet Results ===")
    print(f"  Accuracy (non-neutral): {accuracy:.1%}")
    print(f"  Neutral/abstain count:  {neutral_count} ({neutral_count/len(df):.1%})")
    print(f"  Avg latency:            {avg_latency * 1000:.0f} ms/sample")
    print(f"  Sarcasm detected:       {sarcasm_rate:.1%} of samples")

    return {
        'domain': domain,
        'accuracy': accuracy,
        'neutral_rate': neutral_count / len(df),
        'avg_latency_ms': avg_latency * 1000,
        'sarcasm_rate': sarcasm_rate,
        'n_samples': len(df),
        'n_valid': len(valid),
    }


def get_emotion_distribution(sentic_df: pd.DataFrame) -> pd.Series:
    """
    Count emotion label frequencies. Useful for visualizing emotional profile per domain.
    """
    return (
        sentic_df['emotions']
        .str.upper()
        .value_counts()
        .rename('count')
    )
