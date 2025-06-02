import pandas as pd
import numpy as np
from sklearn.utils import resample
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# ---------- feste Reproduzierbarkeit ----------
SEED = 2025
rng = np.random.RandomState(SEED)
# ---------------------------------------------

# Daten laden
df_cong  = pd.read_csv(Path("bws_congruent.csv"))
df_incong = pd.read_csv(Path("bws_incongruent.csv"))

# Net-Scores pro Person
def netscore(df):
    return (df
            .groupby(["Teilnehmer", "System", "Emotion"])["choice"]
            .sum()
            .reset_index(name="net"))

net_cong  = netscore(df_cong)
net_incong = netscore(df_incong)

from arch.bootstrap import IIDBootstrap

def bootstrap_diff_bca(data1, data2, reps=5000, alpha=0.05, seed=2025):
    bs = IIDBootstrap((data1 - data2).dropna().values, random_state=seed)
    ci_low, ci_high = bs.conf_int(np.mean, reps=reps, method='bca', size=1-alpha).flatten()
    delta_hat = np.mean(data1 - data2)
    signif = ci_low > 0 or ci_high < 0
    return delta_hat, ci_low, ci_high, signif

# Bootstrap-Funktion
def bootstrap_diff_ci(df, emo, sys1, sys2, B=5000, alpha=0.05):
    subset = df.query("Emotion == @emo")
    pivot  = subset.pivot(index="Teilnehmer", columns="System", values="net")
    
    # Punktschätzer (Δ Net-Score)
    delta_hat = pivot[sys1].mean() - pivot[sys2].mean()
    
    diffs = []
    for _ in range(B):
        samp = resample(pivot, replace=True, n_samples=len(pivot), random_state=rng)
        diffs.append(samp[sys1].mean() - samp[sys2].mean())
    
    ci_low, ci_high = np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])
    signif = ci_low > 0 or ci_high < 0
    return delta_hat, ci_low, ci_high, signif

systems  = ["CosyVoice", "EmoSpeech", "EmoKnob", "EmotiVoice"]
emotions = ["Happy", "Sad", "Angry", "Surprised"]

def all_comparisons(df):
    rows = []
    for emo in emotions:
        for s1, s2 in combinations(systems, 2):
            delta, lo, hi, sig = bootstrap_diff_bca(df, emo, s1, s2)
            rows.append({
                "Emotion": emo,
                "System 1": s1,
                "System 2": s2,
                "Δ_Net": delta,
                "CI_low": lo,
                "CI_high": hi,
                "Signifikant": sig
            })
    return pd.DataFrame(rows)

results_cong   = all_comparisons(net_cong)
results_incong = all_comparisons(net_incong)

# Holm‐Korrektur
for res in [results_cong, results_incong]:
    pvals = [0.001 if s else 1 for s in res["Signifikant"]]
    res["p_adjusted"] = multipletests(pvals, method="holm")[1]

# Ausgaben
print("\nBootstrap-Ergebnisse Kongruent:")
print(results_cong)

print("\nBootstrap-Ergebnisse Inkongruent:")
print(results_incong)

# Speichern
results_cong.to_csv("bootstrap_congruent_results.csv", index=False)
results_incong.to_csv("bootstrap_incongruent_results.csv", index=False)