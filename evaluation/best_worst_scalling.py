import pandas as pd
from pathlib import Path

# Load the generated survey data
DATA_PATH = Path("Survey_Entries.csv")
df = pd.read_csv(DATA_PATH)

# Split pairs into separate best and worst columns
q_cols = [c for c in df.columns if c.startswith("Q") and "_best" not in c and "_worst" not in c]
for q in q_cols:
    df[f"{q}_best"] = df[q].str.split(",", n=1).str[0].str.strip().astype(int)
    df[f"{q}_worst"] = df[q].str.split(",", n=1).str[1].str.strip().astype(int)

# Voice system labels
voice_labels = {
    1: "CosyVoice",
    2: "EmoSpeech",
    3: "EmoKnob",
    4: "EmotiVoice"
}

# Emotion mapping for questions
emotion_mapping = {
    "Q1": "Happy (Congruent)",
    "Q2": "Happy (Congruent)",
    "Q3": "Happy (Congruent)",
    "Q4": "Happy (Incongruent)",
    "Q5": "Happy (Incongruent)",
    "Q6": "Happy (Incongruent)",
    "Q7": "Sad (Congruent)",
    "Q8": "Sad (Congruent)",
    "Q9": "Sad (Congruent)",
    "Q10": "Sad (Incongruent)",
    "Q11": "Sad (Incongruent)",
    "Q12": "Sad (Incongruent)",
    "Q13": "Angry (Congruent)",
    "Q14": "Angry (Congruent)",
    "Q15": "Angry (Congruent)",
    "Q16": "Angry (Incongruent)",
    "Q17": "Angry (Incongruent)",
    "Q18": "Angry (Incongruent)",
    "Q19": "Surprised (Congruent)",
    "Q20": "Surprised (Congruent)",
    "Q21": "Surprised (Congruent)",
    "Q22": "Surprised (Incongruent)",
    "Q23": "Surprised (Incongruent)",
    "Q24": "Surprised (Incongruent)"
}

# Emotion groups
emotion_groups = {
    "Happy Congruent": ["Q1", "Q2", "Q3"],
    "Happy Incongruent": ["Q4", "Q5", "Q6"],
    "Sad Congruent": ["Q7", "Q8", "Q9"],
    "Sad Incongruent": ["Q10", "Q11", "Q12"],
    "Angry Congruent": ["Q13", "Q14", "Q15"],
    "Angry Incongruent": ["Q16", "Q17", "Q18"],
    "Surprised Congruent": ["Q19", "Q20", "Q21"],
    "Surprised Incongruent": ["Q22", "Q23", "Q24"]
}

output_file = Path("best_worst_scalling.txt")

with open(output_file, "w") as file:
    # Individuelle Fragen-Ergebnisse
    for q in q_cols:
        best_counts = df[f"{q}_best"].value_counts().sort_index()
        worst_counts = df[f"{q}_worst"].value_counts().sort_index()

        best_counts = best_counts.reindex(range(1, 5), fill_value=0)
        worst_counts = worst_counts.reindex(range(1, 5), fill_value=0)

        net_scores = best_counts - worst_counts 

        file.write(f"{q} - {emotion_mapping[q]}\n")
        file.write("Voice System\tBest - Worst (Net Score)\n")
        for i in range(1, 5):
            file.write(f"{voice_labels[i]}\t{net_scores[i]}\n")
        file.write("\n")

    # Aggregierte Emotionsergebnisse
    for group_name, questions in emotion_groups.items():
        total_best = pd.Series(0, index=range(1, 5))
        total_worst = pd.Series(0, index=range(1, 5))
 
        for q in questions:
            best_counts = df[f"{q}_best"].value_counts().sort_index()
            worst_counts = df[f"{q}_worst"].value_counts().sort_index()

            best_counts = best_counts.reindex(range(1, 5), fill_value=0)
            worst_counts = worst_counts.reindex(range(1, 5), fill_value=0)

            total_best += best_counts
            total_worst += worst_counts

        net_scores = total_best - total_worst

        file.write(f"{group_name} - Aggregated MaxDiff Net Scores\n")
        file.write("Voice System\tBest - Worst (Net Score)\n")
        for i in range(1, 5):
            file.write(f"{voice_labels[i]}\t{net_scores[i]}\n")
        file.write("\n")

print("All results have been saved to best_worst_scalling.txt.")
