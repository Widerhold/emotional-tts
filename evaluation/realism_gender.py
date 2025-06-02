import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import weightstats as smws
from pingouin import compute_effsize, welch_anova, ttest

# ------------------------------------------------------------------
# Daten laden und aufbereiten
# ------------------------------------------------------------------
df = pd.read_csv("survey_entries.csv")
df["Realismus"] = pd.to_numeric(df["Realismus"], errors="coerce")

geschlecht_map = {1: "männlich", 2: "weiblich", 3: "divers"}
df["Geschlecht"] = df["Geschlecht"].map(geschlecht_map)

m = df.loc[df.Geschlecht == "männlich", "Realismus"].dropna()
w = df.loc[df.Geschlecht == "weiblich", "Realismus"].dropna()

# ------------------------------------------------------------------
# Deskriptivstatistik
# ------------------------------------------------------------------
def descriptives(series):
    ci = stats.t.interval(
        0.95, len(series)-1, loc=series.mean(), scale=stats.sem(series)
    )
    return {
        "N": len(series),
        "M": series.mean(),
        "SD": series.std(ddof=1),
        "95% CI": ci
    }

desc_m, desc_w = descriptives(m), descriptives(w)
print("Deskriptivwerte")
print(pd.DataFrame([desc_m, desc_w], index=["Männer", "Frauen"]), "\n")

# ------------------------------------------------------------------
# Vorbedingungen
# ------------------------------------------------------------------
print("Shapiro–Wilk (Normalität):")
print("  Männer :", stats.shapiro(m).pvalue)
print("  Frauen :", stats.shapiro(w).pvalue)
print("Levene (Varianzgleichheit):", stats.levene(m, w).pvalue, "\n")

# ------------------------------------------------------------------
# Welch-t-Test
# ------------------------------------------------------------------
t_stat, p_val = stats.ttest_ind(m, w, equal_var=False)
df_welch = welch_anova(dv="Realismus", between="Geschlecht", data=df)
print(f"Welch-t: t = {t_stat:.2f}, p = {p_val:.3f}")
print("Welch-df laut Pingouin:", float(df_welch['ddof2']), "\n")

# Effektstärke
d = compute_effsize(m, w, eftype='cohen')
print(f"Cohen’s d = {d:.2f}  (klein ≈ .20, mittel ≈ .50, groß ≈ .80)\n")

# ------------------------------------------------------------------
# Non-parametrischer Vergleich
# ------------------------------------------------------------------
u, p_u = stats.mannwhitneyu(m, w, alternative="two-sided")
print(f"Mann-Whitney-U: U = {u}, p = {p_u:.3f}\n")

# ------------------------------------------------------------------
# Äquivalenztest (TOST, ±0.30 SD)
# ------------------------------------------------------------------
low, high = -0.30, 0.30
p_low, p_high, _ = smws.ttost_ind(m, w, low, high, usevar='unequal')

# Extrahieren der p-Werte aus den Tupeln
p_low_value = p_low[0] if isinstance(p_low, tuple) else p_low
p_high_value = p_high[0] if isinstance(p_high, tuple) else p_high

print("TOST (Δ = ±0,30 SD):")
print(f"  p_low  = {p_low_value:.3f}, p_high = {p_high_value:.3f}\n")



# ------------------------------------------------------------------
# Bayes-t-Test (Cauchy r = 0.707)
# ------------------------------------------------------------------
from pingouin import ttest, bayesfactor_ttest

# Standard t-Test mit Welch-Korrektur
ttest_results = ttest(m, w, correction=True)
print(ttest_results)

# Bayes-Faktor separat berechnen (default prior r = 0.707)
bf10 = bayesfactor_ttest(ttest_results['T'].iloc[0], nx=len(m), ny=len(w), paired=False)

print(f"\nBayes-Faktor (BF10) = {bf10:.3f}")
