import pandas as pd
from pathlib import Path

DATA_PATH = Path("Survey_Entries.csv")
df = pd.read_csv(DATA_PATH)

q_cols = [c for c in df.columns if c.startswith("Q") and "_best" not in c and "_worst" not in c]
for q in q_cols:
    df[f"{q}_best"] = df[q].str.split(",", n=1).str[0].str.strip().astype(int)
    df[f"{q}_worst"] = df[q].str.split(",", n=1).str[1].str.strip().astype(int)

voice_labels = {1: "CosyVoice", 2: "EmoSpeech", 3: "EmoKnob", 4: "EmotiVoice"}

emotion_groups = {
    "Happy": ["Q1", "Q2", "Q3"],
    "Sad": ["Q7", "Q8", "Q9"],
    "Angry": ["Q13", "Q14", "Q15"],
    "Surprised": ["Q19", "Q20", "Q21"]
}

demographics = {
    "Geschlecht": {1: "Male", 2: "Female", 3: "Diverse"},
    "Altersgruppe": {1: "<30", 2: "30-44", 3: "45-59", 4: ">60"},
    "Englischkenntnisse": {1: "A1-A2", 2: "B1-B2", 3: "C1-C2"}
}

output_file = Path("demographic_bws_raw_results.txt")

def calculate_raw_scores(subset_df, questions):
    raw_scores = pd.Series(0, index=range(1, 5))
    for q in questions:
        for _, row in subset_df.iterrows():
            raw_scores[row[f"{q}_best"]] += 1
            raw_scores[row[f"{q}_worst"]] -= 1
    return raw_scores

# def calculate_raw_scores(subset_df, questions):
#     raw_scores = pd.Series(0, index=range(1, 5))
#     half = len(questions) // 2              # erste drei Fragen kongruent, zweite drei inkongruent
#     for idx, q in enumerate(questions):
#         sign = 1 if idx < half else -1      # Vorzeichen umdrehen, wenn inkongruent
#         for _, row in subset_df.iterrows():
#             raw_scores[row[f"{q}_best"]]  += sign
#             raw_scores[row[f"{q}_worst"]] -= sign
#     return raw_scores

with open(output_file, "w") as file:
    for demo_col, labels in demographics.items():
        file.write(f"Demographic Analysis by {demo_col} (Raw Best-Worst Scores)\n")
        file.write("===========================================================\n\n")

        for demo_value, label in labels.items():
            subset_df = df[df[demo_col] == demo_value]
            n_samples = len(subset_df)
            file.write(f"{demo_col}: {label} (N={n_samples})\n\n")

            overall_scores = pd.Series(0, index=range(1, 5))

            for emotion, questions in emotion_groups.items():
                raw_scores = calculate_raw_scores(subset_df, questions)
                overall_scores += raw_scores

                file.write(f"{emotion} Emotion Raw Scores:\n")
                file.write("Voice System\tScore\n")
                for i in range(1, 5):
                    file.write(f"{voice_labels[i]}\t{raw_scores[i]}\n")
                file.write("\n")

            file.write("Overall Raw Scores per Voice System:\n")
            file.write("Voice System\tOverall Score\n")
            for i in range(1, 5):
                file.write(f"{voice_labels[i]}\t{overall_scores[i]}\n")
            file.write("\n")

            file.write("-----------------------------------\n")
        file.write("\n")

print("Demographic analysis results saved to demographic_bws_raw_results.txt.")
