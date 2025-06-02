#!/usr/bin/env python3
"""
Analyse des Realismusgrads nach Altersgruppen

Dieses Skript lädt die Daten aus survey_entries.csv
berechnet deskriptive Kennwerte
prüft die Voraussetzungen der ANOVA
führt je nach Ergebnis eine klassische ANOVA oder eine Welch ANOVA durch
fügt einen Kruskal Wallis Test als robuste Ergänzung hinzu
und berechnet Cliff δ für alle Gruppenpaare.
"""

import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

# Laufzeitwarnungen unterdrücken damit der Output übersichtlich bleibt
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- Daten laden ----------
DATA_PATH = "survey_entries.csv"
data = pd.read_csv(DATA_PATH)

# Spalte Realismus als numerisch sicherstellen
data["Realismus"] = pd.to_numeric(data["Realismus"], errors="coerce")

# Altersgruppe kodieren
age_map = {1: "<30", 2: "30-44", 3: "45-59", 4: ">60"}
data["Altersgruppe"] = data["Altersgruppe"].map(age_map)

# Gruppen als Dictionary zusammenstellen
groups = {g: d["Realismus"].dropna() for g, d in data.groupby("Altersgruppe")}

# ---------- Deskriptive Kennwerte ----------

def descriptive_stats(series: pd.Series):
    n = len(series)
    mean = series.mean()
    sd = series.std(ddof=1)
    median = series.median()
    iqr = stats.iqr(series, nan_policy="omit")
    if n >= 3:
        ci_low, ci_high = stats.t.interval(
            0.95, n - 1, loc=mean, scale=sd / np.sqrt(n)
        )
    else:
        ci_low, ci_high = np.nan, np.nan
    return n, mean, sd, median, iqr, ci_low, ci_high

rows = [
    (g, *descriptive_stats(vals)) for g, vals in sorted(groups.items())
]

desc_df = pd.DataFrame(
    rows,
    columns=[
        "Gruppe",
        "N",
        "Mittelwert",
        "SD",
        "Median",
        "IQR",
        "CI_low",
        "CI_high",
    ],
)
print("\nDeskriptive Statistiken")
print(desc_df.round(2).to_string(index=False))

# ---------- Annahmen prüfen ----------
shapiro = {
    g: stats.shapiro(vals)[1]
    for g, vals in groups.items()
    if len(vals) > 2
}
levene_p = stats.levene(*groups.values(), center="median")[1]
print("\nShapiro-Wilk p-Werte:", {k: round(v, 3) for k, v in shapiro.items()})
print("Levene p-Wert:", round(levene_p, 3))

norm_ok = all(p > 0.05 for p in shapiro.values())
var_ok = levene_p > 0.05

# ---------- Omnibus Tests ----------
if norm_ok and var_ok:
    f_stat, p_val = stats.f_oneway(*groups.values())
    eta2 = pg.anova(data=data, dv="Realismus", between="Altersgruppe")[
        "eta-square"
    ][0]
    print(
        f"\nKlassische ANOVA   F = {f_stat:.3f}   p = {p_val:.3f}   η² = {eta2:.3f}"
    )
else:
    welch = pg.welch_anova(data=data, dv="Realismus", between="Altersgruppe")
    f_stat = welch.loc[0, "F"]
    p_val = welch.loc[0, "p-unc"]
    omega2 = welch.loc[0, "np2"]
    print(
        f"\nWelch ANOVA        F = {f_stat:.3f}   p = {p_val:.3f}   ω² = {omega2:.3f}"
    )

# Kruskal Wallis als robuste Alternative
H, p_kw = stats.kruskal(*groups.values())
print(f"Kruskal Wallis      H = {H:.3f}   p = {p_kw:.3f}")

# ---------- Cliff δ ----------

def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    """
    Berechnet Cliff δ (gleichbedeutend mit Rank-Biserial)
    """
    nx = len(x)
    ny = len(y)
    gt = sum(1 for xi, yi in product(x, y) if xi > yi)
    lt = sum(1 for xi, yi in product(x, y) if xi < yi)
    return (gt - lt) / (nx * ny)

delta_rows = []
for a, b in combinations(sorted(groups.keys()), 2):
    d = cliffs_delta(groups[a], groups[b])
    delta_rows.append((a, b, d))

delta_df = pd.DataFrame(delta_rows, columns=["A", "B", "Cliff_delta"])
print("\nCliff δ pro Paar")
print(delta_df.round(3).to_string(index=False))
