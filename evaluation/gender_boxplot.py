import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Daten laden & säubern
# ------------------------------------------------------------------
df = pd.read_csv("survey_entries.csv")
df["Realismus"] = pd.to_numeric(df["Realismus"], errors="coerce")

geschlecht_map = {1: "männlich", 2: "weiblich", 3: "divers"}
df["Geschlecht"] = df["Geschlecht"].map(geschlecht_map)

# Daten nach Geschlecht aufteilen
m = df.loc[df.Geschlecht == "männlich", "Realismus"].dropna()
w = df.loc[df.Geschlecht == "weiblich", "Realismus"].dropna()

# ------------------------------------------------------------------
# Boxplot rendern mit Matplotlib
# ------------------------------------------------------------------
plt.figure(figsize=(5, 3))
box = plt.boxplot([m, w], labels=["Männlich\n(N={})".format(len(m)), "Weiblich\n(N={})".format(len(w))], patch_artist=True)

for patch in box['boxes']:
    patch.set_facecolor('#B1B3EB')

plt.ylabel('Realismusgrad (1–5)')
plt.ylim(1, 5)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
#plt.title('Realismusbewertungen nach Geschlecht')

plt.tight_layout()
plt.savefig("gender_realism_boxplot.png", dpi=1200)
plt.show()
