import pandas as pd
from pathlib import Path

DATA_PATH = Path("Survey_Entries.csv")
df_raw = pd.read_csv(DATA_PATH)

# Best- und Worst-Spalten anlegen
q_cols = [c for c in df_raw.columns if c.startswith("Q") and "_best" not in c and "_worst" not in c]
for q in q_cols:
    best, worst = f"{q}_best", f"{q}_worst"
    df_raw[best] = df_raw[q].str.split(",", n=1).str[0].str.strip().astype(int)
    df_raw[worst] = df_raw[q].str.split(",", n=1).str[1].str.strip().astype(int)

voice_labels = {
    1: "CosyVoice",
    2: "EmoSpeech",
    3: "EmoKnob",
    4: "EmotiVoice"
}

emotion_lookup = {
    "Q1": ("Happy", "Congruent"),  "Q2": ("Happy", "Congruent"),  "Q3": ("Happy", "Congruent"),
    "Q4": ("Happy", "Incongruent"),"Q5": ("Happy", "Incongruent"),"Q6": ("Happy", "Incongruent"),
    "Q7": ("Sad", "Congruent"),    "Q8": ("Sad", "Congruent"),    "Q9": ("Sad", "Congruent"),
    "Q10": ("Sad", "Incongruent"), "Q11": ("Sad", "Incongruent"), "Q12": ("Sad", "Incongruent"),
    "Q13": ("Angry", "Congruent"), "Q14": ("Angry", "Congruent"), "Q15": ("Angry", "Congruent"),
    "Q16": ("Angry", "Incongruent"),"Q17": ("Angry", "Incongruent"),"Q18": ("Angry", "Incongruent"),
    "Q19": ("Surprised", "Congruent"),"Q20": ("Surprised", "Congruent"),"Q21": ("Surprised", "Congruent"),
    "Q22": ("Surprised", "Incongruent"),"Q23": ("Surprised", "Incongruent"),"Q24": ("Surprised", "Incongruent")
}

long_rows = []
for idx, row in df_raw.iterrows():
    teilnehmer = idx
    for q in q_cols:
        emo, cong = emotion_lookup[q]
        # Best-Entscheidung
        long_rows.append({
            "Teilnehmer": teilnehmer,
            "Item": q,
            "System": voice_labels[row[f"{q}_best"]],
            "Emotion": emo,
            "Kongruenz": cong,
            "choice": 1
        })
        # Worst-Entscheidung
        long_rows.append({
            "Teilnehmer": teilnehmer,
            "Item": q,
            "System": voice_labels[row[f"{q}_worst"]],
            "Emotion": emo,
            "Kongruenz": cong,
            "choice": 0
        })

df_long = pd.DataFrame(long_rows)

# Ergebnisdateien
df_long.to_csv("bws_long.csv", index=False)                 # gesamter Datensatz
df_long.query("Kongruenz == 'Congruent'").to_csv("bws_congruent.csv", index=False)
df_long.query("Kongruenz == 'Incongruent'").to_csv("bws_incongruent.csv", index=False)
