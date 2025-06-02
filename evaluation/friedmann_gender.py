# -----------------------------------------------------------
# Friedman-Tests (Monte-Carlo) nach Geschlecht × Kongruenz × Emotion
#   • 4 TTS-Systeme → Rang-Daten (Best = +1, Worst = –1)
#   • 10 000 Zufalls­permutationen, fester Seed
# -----------------------------------------------------------

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# ---------- Konfiguration ----------
DATA_PATH = Path("Survey_Entries.csv")

voice_labels = {1: "CosyVoice", 2: "EmoSpeech", 3: "EmoKnob", 4: "EmotiVoice"}

gender_map = {1: "Male", 2: "Female", 3: "Diverse"}

emotion_questions = {
    "Happy":     ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
    "Sad":       ["Q7", "Q8", "Q9", "Q10", "Q11", "Q12"],
    "Angry":     ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Surprised": ["Q19", "Q20", "Q21", "Q22", "Q23", "Q24"],
}

MC_PERMUTATIONS = 10_000
SEED = 2025
rng = np.random.default_rng(SEED)           
np.random.seed(SEED)

# ---------- Daten laden & vorbereiten ----------
df = pd.read_csv(DATA_PATH)

# Best/Worst-Spalten erzeugen
base_qs = [c for c in df.columns if c.startswith("Q") and "_best" not in c]
for q in base_qs:
    df[f"{q}_best"]  = df[q].str.split(",", n=1).str[0].str.strip().astype(int)
    df[f"{q}_worst"] = df[q].str.split(",", n=1).str[1].str.strip().astype(int)


# ---------- Hilfsfunktionen ----------
def calc_scores(row: pd.Series, qs: list[str]) -> pd.Series:
    scores = pd.Series(0, index=range(1, 5))
    for q in qs:
        scores[row[f"{q}_best"]]  += 1
        scores[row[f"{q}_worst"]] -= 1
    return scores

def kendalls_w(chi2: float, n: int, k: int) -> float:
    return chi2 / (n * (k - 1)) if n else np.nan

def tie_stats(df_block: pd.DataFrame) -> tuple[int, float]:
    k = df_block.shape[1]
    tie_rows = df_block.apply(lambda r: len(set(r)) < k, axis=1)
    return int(tie_rows.sum()), 100 * tie_rows.mean()

def friedman_mc(data: np.ndarray, n_perm: int = MC_PERMUTATIONS) -> float:
    obs = friedmanchisquare(*data.T).statistic
    ge = (friedmanchisquare(*rng.permuted(data, axis=1).T).statistic >= obs
          for _ in range(n_perm))
    count = sum(ge)
    return (count + 1) / (n_perm + 1)     # unbiased

# ---------- Long-Format-Tabelle ----------
rows = []
for pid, row in df.iterrows():
    age = gender_map.get(row["Geschlecht"])
    if age is None:
        continue
    for emotion, qs in emotion_questions.items():
        cong_qs, incong_qs = qs[:3], qs[3:]
        for label, sel_qs in [("Kongruent", cong_qs), ("Inkongruent", incong_qs)]:
            scores = calc_scores(row, sel_qs)
            for sys_id, score in scores.items():
                rows.append(
                    {
                        "Participant":  pid,
                        "Geschlecht": age,
                        "Kongruenz":    label,
                        "Emotion":      emotion,
                        "System":       voice_labels[sys_id],
                        "Score":        score,
                    }
                )

scores_df = pd.DataFrame(rows)

# ---------- Friedman-Loops ----------
for age in scores_df["Geschlecht"].unique():
    for cong in ["Kongruent", "Inkongruent"]:
        subset = scores_df.query("Geschlecht == @age & Kongruenz == @cong")
        if subset.empty:
            continue
        pivot = (subset
                 .pivot_table(index=["Participant", "Emotion"],
                              columns="System",
                              values="Score")
                 .dropna())

        print(f"\nGeschlecht: {age}  |  {cong}")
        for emotion in pivot.index.get_level_values("Emotion").unique():
            emo_data = pivot.xs(emotion, level="Emotion").to_numpy(int)
            n, k = emo_data.shape
            chi2, p_asymp = friedmanchisquare(*emo_data.T)
            ties_n, ties_pct = tie_stats(pivot.xs(emotion, level="Emotion"))

            use_mc = (n < 10) or (ties_pct > 50)
            p_final = friedman_mc(emo_data) if use_mc else p_asymp
            note = "Monte-Carlo" if use_mc else "asymptotisch"
            W = kendalls_w(chi2, n, k)

            print(f"  Emotion: {emotion:<9} χ²({k-1}) = {chi2:.3f}, "
                  f"p = {p_final:.4f} ({note}), W = {W:.3f}, "
                  f"Ties: {ties_n}/{n} ({ties_pct:.1f} %)")

            if p_final < 0.05:
                melted = (pivot.xs(emotion, level="Emotion")
                               .reset_index()
                               .melt(id_vars="Participant",
                                     var_name="System",
                                     value_name="Score"))
                dunn = sp.posthoc_dunn(melted,
                                        val_col="Score",
                                        group_col="System",
                                        p_adjust="bonferroni")
                sig_pairs = [(a, b, dunn.loc[a, b])
                             for a in dunn.index
                             for b in dunn.columns
                             if a < b and dunn.loc[a, b] < 0.05]
                if sig_pairs:
                    print("    Signifikante Paare:")
                    for a, b, p in sig_pairs:
                        print(f"      - {a} vs {b}: p = {p:.4f}")
                else:
                    print("    Keine signifikanten Paare")
