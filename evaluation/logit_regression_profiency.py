# -----------------------------------------------------------
# Binomiale gemischte Logit-Regression (GEE)
#   • Fixed effect  : Englischkenntnisse
#   • Random effect : Participant (cluster / exchangeable)
#   • Outcome       : best  (1 = System wurde als „best“ gewählt)
# -----------------------------------------------------------

import pandas as pd
from pathlib import Path
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

# ---------- Einstellungen ----------
DATA_PATH = Path("Survey_Entries.csv")

voice_labels = {1: "CosyVoice", 2: "EmoSpeech", 3: "EmoKnob", 4: "EmotiVoice"}
emotion_groups = {
    "Happy":     ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
    "Sad":       ["Q7", "Q8", "Q9", "Q10", "Q11", "Q12"],
    "Angry":     ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Surprised": ["Q19", "Q20", "Q21", "Q22", "Q23", "Q24"]
}

# Mapping Alterscode → Label
profiency_map = {1: "A1-A2", 2: "B1-B2", 3: "C1-C2"}

# ---------- Daten einlesen und splitten ----------
df_raw = pd.read_csv(DATA_PATH)

for q in [c for c in df_raw.columns if c.startswith("Q") and "_best" not in c]:
    df_raw[f"{q}_best"]  = df_raw[q].str.split(",", n=1).str[0].astype(int)
    df_raw[f"{q}_worst"] = df_raw[q].str.split(",", n=1).str[1].astype(int)

# ---------- Long-Format: jede Entscheidung = 1 Zeile ----------
rows = []
for pid, row in df_raw.iterrows():
    age_label = profiency_map.get(row["Englischkenntnisse"])
    if age_label is None:           # unbekannter Code
        continue
    for emotion, qs in emotion_groups.items():
        for q in qs:
            best_sys = row[f"{q}_best"]
            for sys_id, sys_name in voice_labels.items():
                rows.append(
                    {
                        "Participant":  pid,
                        "Englischkenntnisse": age_label,
                        "Emotion":      emotion,
                        "System":       sys_name,
                        "best":         int(best_sys == sys_id)
                    }
                )

long_df = pd.DataFrame(rows)

# ---------- Modell-Loop ----------
results = []
skip_counter = 0

for emotion in emotion_groups:
    for sys in voice_labels.values():
        sub = long_df[(long_df["Emotion"] == emotion) &
                      (long_df["System"]  == sys)]

        # überspringen, wenn nur eine Englischkenntnisse vertreten
        if sub["Englischkenntnisse"].nunique() < 2:
            skip_counter += 1
            continue

        model = GEE.from_formula(
            "best ~ C(Englischkenntnisse)",
            groups="Participant",
            data=sub,
            family=Binomial(),
            cov_struct=Exchangeable()
        )
        gee_res = model.fit()

        for pname in [p for p in gee_res.params.index
                      if p.startswith("C(Englischkenntnisse)[T.")]:
            coef   = gee_res.params[pname]
            pval   = gee_res.pvalues[pname]
            ci_low, ci_high = gee_res.conf_int().loc[pname]
            results.append([emotion, sys, pname, coef, pval, ci_low, ci_high])

# ---------- Multiple-Test-Korrektur ----------
res_df = pd.DataFrame(results,
                      columns=["Emotion", "System", "Contrast",
                               "β", "p_raw", "CI_low", "CI_high"])

if not res_df.empty:
    res_df["p_adj"] = multipletests(res_df["p_raw"], method="holm")[1]

    # ---------- Ausgaben ----------
    print(f"\nTests insgesamt durchgeführt: {len(res_df)}")
    print(f"Kombinationen übersprungen (nur eine Englischkenntnisse präsent): {skip_counter}\n")

    sig = res_df[res_df["p_adj"] < 0.05]
    if sig.empty:
        print("Keine Holm-signifikanten Englischkenntnisseneffekte gefunden.")
    else:
        print("Signifikante Englischkenntnisseneffekte (Holm-adjustiert p < 0.05):")
        for _, r in sig.iterrows():
            print(f"{r.Emotion:<10} | {r.System:<10} | {r.Contrast:<18} "
                  f"β = {r.β:+.3f} (95% CI {r.CI_low:.2f}…{r.CI_high:.2f}) | "
                  f"p_adj = {r.p_adj:.4f}")
else:
    print("Es wurden keine gültigen Tests durchgeführt.")
