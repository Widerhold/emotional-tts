import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import numpy as np

DATA_PATH = Path("Survey_Entries.csv")
df = pd.read_csv(DATA_PATH)


# Aufsplitten in best / worst
for q in [c for c in df.columns if c.startswith("Q") and "_best" not in c]:
    df[f"{q}_best"]  = df[q].str.split(",", n=1).str[0].astype(int)
    df[f"{q}_worst"] = df[q].str.split(",", n=1).str[1].astype(int)

voice_labels = {1: "CosyVoice", 2: "EmoSpeech", 3: "EmoKnob", 4: "EmotiVoice"}
emotion_groups = {
    "Happy": ["Q1","Q2","Q3","Q4","Q5","Q6"],
    "Sad":   ["Q7","Q8","Q9","Q10","Q11","Q12"],
    "Angry": ["Q13","Q14","Q15","Q16","Q17","Q18"],
    "Surprised": ["Q19","Q20","Q21","Q22","Q23","Q24"]
}
gender_map = {1: "Male", 2: "Female"}

# Häufigkeits­tabellen + Test
records = []
for emotion, qs in emotion_groups.items():
    for sys_id, sys_name in voice_labels.items():
        tbl = np.zeros((2,2), dtype=int)   # rows: best/worst, cols: male/female
        
        for _, row in df.iterrows():
            g = gender_map.get(row["Geschlecht"], None)
            if g is None: continue
            for q in qs:
                if row[f"{q}_best"]  == sys_id: tbl[0, 0 if g=="Male" else 1] += 1
                if row[f"{q}_worst"] == sys_id: tbl[1, 0 if g=="Male" else 1] += 1

        # Wahl des passenden Tests
        chi2, p, dof, exp = chi2_contingency(tbl, correction=True)
        test_used = "chi2"
        if (exp < 5).any():               # kleine erwartete Zellen
            _, p = fisher_exact(tbl)      # zweiseitig
            test_used = "fisher"
            chi2 = np.nan

        # Effektgröße (nur für chi² sinnvoll)
        cramer_v = np.nan
        if test_used == "chi2":
            n = tbl.sum()
            cramer_v = np.sqrt(chi2 / n)

        records.append([emotion, sys_name, test_used, chi2, cramer_v, p])

# Alpha-Kontrolle (Holm)
df_out = pd.DataFrame(records, columns=["Emotion","System","Test","Chi2",
                                        "CramersV","p_raw"])
df_out["p_adj"] = multipletests(df_out["p_raw"], method="holm")[1]

# Ausgabe
for _, r in df_out.iterrows():
    print(f"{r.Emotion:10} | {r.System:10} | {r.Test:6} | "
          f"p_adj = {r.p_adj:.4f}"
          f"{'  *sig*' if r.p_adj < 0.05 else ''}")
