import pandas as pd
import numpy as np
from pathlib import Path

results = Path('results')

bert = {d: pd.read_csv(results/f'bert_{d}.csv') for d in ['imdb','twitter','amazon']}
llm  = {d: pd.read_csv(results/f'llm_{d}.csv')  for d in ['imdb','twitter','amazon']}
sent = {d: pd.read_csv(results/f'sentic_{d}.csv') for d in ['imdb','twitter','amazon']}

print('=== BERT CALIBRATION GAPS ===')
for d in ['imdb','twitter','amazon']:
    df = bert[d]
    corr  = df[df['correct']==True]['bert_confidence'].mean()
    wrong = df[df['correct']==False]['bert_confidence'].mean()
    print(f'{d}: correct={corr:.3f}  wrong={wrong:.3f}  gap={corr-wrong:.3f}')

print()
print('=== LLM ACCURACY + GENERALIZATION ===')
llm_accs = {}
for d in ['imdb','twitter','amazon']:
    df = llm[d]
    valid = df[df['llm_pred']!=-1]
    acc = (valid['llm_pred'] == valid['ground_truth']).mean()
    llm_accs[d] = acc
    print(f'{d}: {acc:.3f} ({acc:.1%})')
print(f'Gap: {max(llm_accs.values()):.3f} - {min(llm_accs.values()):.3f} = {max(llm_accs.values())-min(llm_accs.values()):.3f}')

print()
print('=== BERT ACCURACY + GENERALIZATION ===')
bert_accs = {}
for d in ['imdb','twitter','amazon']:
    acc = bert[d]['correct'].mean()
    bert_accs[d] = acc
    print(f'{d}: {acc:.3f} ({acc:.1%})')
print(f'Gap: {max(bert_accs.values()):.3f} - {min(bert_accs.values()):.3f} = {max(bert_accs.values())-min(bert_accs.values()):.3f}')

print()
print('=== BERT vs LLM per domain (delta) ===')
for d in ['imdb','twitter','amazon']:
    b = bert_accs[d]
    l = llm_accs[d]
    print(f'{d}: BERT={b:.1%}  LLM={l:.1%}  LLM_advantage={l-b:+.1%}')

print()
print('=== SENTIC ACCURACY ===')
for d in ['imdb','twitter','amazon']:
    df = sent[d]
    valid = df[df['sentic_pred']!=-1]
    acc = (valid['sentic_pred'] == valid['ground_truth']).mean()
    neutral_rate = (df['sentic_pred']==-1).mean()
    print(f'{d}: acc={acc:.1%}  neutral_rate={neutral_rate:.1%}')

print()
print('=== BERT FAILURE SAMPLES (IMDB - first 3) ===')
df = bert['imdb']
fails = df[df['correct']==False].head(3)
for _, r in fails.iterrows():
    gt   = 'POS' if r['ground_truth']==1 else 'NEG'
    pred = 'POS' if r['bert_pred']==1 else 'NEG'
    conf = r['bert_confidence']
    wc   = r['word_count']
    txt  = str(r['text'])[:180]
    print(f'GT:{gt} BERT:{pred} conf:{conf:.3f} words:{wc}')
    print(f'  {txt}')
    print()

print('=== BERT FAILURE SAMPLES (TWITTER - first 3) ===')
df = bert['twitter']
fails = df[df['correct']==False].head(3)
for _, r in fails.iterrows():
    gt   = 'POS' if r['ground_truth']==1 else 'NEG'
    pred = 'POS' if r['bert_pred']==1 else 'NEG'
    conf = r['bert_confidence']
    wc   = r['word_count']
    txt  = str(r['text'])[:180]
    print(f'GT:{gt} BERT:{pred} conf:{conf:.3f} words:{wc}')
    print(f'  {txt}')
    print()

print('=== LLM FAILURE SAMPLES (AMAZON - first 3) ===')
df = llm['amazon']
fails = df[(df['correct']==False) & (df['llm_pred']!=-1)].head(3)
for _, r in fails.iterrows():
    gt   = 'POS' if r['ground_truth']==1 else 'NEG'
    pred = 'POS' if r['llm_pred']==1 else 'NEG'
    raw  = r['llm_label_raw']
    wc   = r['word_count']
    txt  = str(r['text'])[:180]
    print(f'GT:{gt} LLM:{pred} raw:{raw} words:{wc}')
    print(f'  {txt}')
    print()

print('=== THREE-WAY DISAGREEMENTS ===')
for d in ['imdb','twitter','amazon']:
    n = min(len(bert[d]), len(llm[d]), len(sent[d]))
    b = bert[d]['bert_pred'].values[:n]
    l = llm[d]['llm_pred'].values[:n]
    s = sent[d]['sentic_pred'].values[:n]
    valid = (l != -1) & (s != -1)
    all_diff = valid & (b != l) & (l != s) & (b != s)
    pct = all_diff.sum()/valid.sum()
    print(f'{d}: {int(all_diff.sum())} three-way disagreements out of {int(valid.sum())} valid ({pct:.1%})')

print()
print('=== SARCASM CROSS-ANALYSIS (BERT accuracy on sarcastic samples) ===')
for d in ['imdb','twitter','amazon']:
    n = min(len(bert[d]), len(sent[d]))
    sdf = sent[d].head(n)
    bdf = bert[d].head(n)
    sarc = sdf['is_sarcastic'].fillna(False).astype(bool)
    n_sarc = sarc.sum()
    bert_overall = bdf['correct'].mean()
    if n_sarc > 0:
        gt_sarc   = sdf['ground_truth'].values[sarc]
        pred_sarc = bdf['bert_pred'].values[sarc]
        bert_sarc = (pred_sarc == gt_sarc).mean()
        print(f'{d}: n_sarcastic={n_sarc}  BERT_overall={bert_overall:.1%}  BERT_on_sarcastic={bert_sarc:.1%}  delta={bert_sarc-bert_overall:+.1%}')
    else:
        print(f'{d}: no sarcastic samples detected  BERT_overall={bert_overall:.1%}')

print()
print('=== EMOTION DISTRIBUTION (top 5 per domain) ===')
for d in ['imdb','twitter','amazon']:
    df = sent[d]
    top = df['emotions'].str.upper().value_counts().head(5)
    print(f'{d.upper()}: {dict(top)}')
