import pandas as pd
from math import sqrt
from scipy.stats import wilcoxon, norm, shapiro

realism = pd.read_csv("survey_entries.csv")["Realismus"].astype(float).dropna()
n = len(realism)

# Wilcoxon-Test (gegen Median = 3)
W, p = wilcoxon(realism - 3, zero_method="wilcox", correction=True, alternative="two-sided")

# z-Wert aus p rekonstruieren
z = norm.ppf(p / 2) * -1  # zwei­seitig → /2, Vorzeichen umkehren
r = z / sqrt(n)           # Effektgröße nach Rosenthal

print(f"n = {n},  Mittel = {realism.mean():.2f},  SD = {realism.std(ddof=1):.2f}")
print("Shapiro-p =", shapiro(realism).pvalue)  # Non-Normalität bestätigt
print(f"Wilcoxon: W = {W:.2f}, z = {z:.2f}, p = {p:.3f}, r = {r:.2f}")
