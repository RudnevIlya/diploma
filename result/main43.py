import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Загрузка данных ===
df = pd.read_csv("final_merged_output_with_bert.csv")
df = df[df["Rating"] >= 1]  # убираем мусор

# === 2. Классификация рейтингов ===
class_ranges = {
    "низкий": (0, 3.85),
    "средний": (3.85, 4.25),
    "высокий": (4.25, 5.01)
}
class_names = list(class_ranges.keys())

def rating_to_class(r):
    for idx, (name, (lo, hi)) in enumerate(class_ranges.items()):
        if lo <= r < hi:
            return idx
    return len(class_ranges) - 1

df["Class"] = df["Rating"].apply(rating_to_class)

# === 3. Признаки для анализа ===
features = [
"HD_D", "Average_TFIDF", "Emotional", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio",
    "SimLow", "SimMid", "SimHigh"
]
# features = [
# "Emotional",
#     "AvgWordLength", "AvgSentenceLength",
#     "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","AbstractNounRatio",
#     "SimLow", "SimMid", "SimHigh"
# ]
df = df.dropna(subset=features + ["Class"])

# === 4. Распределение классов ===
print("🔢 Распределение классов:", Counter(df["Class"]))

# === 5. Boxplot сравнение признаков по классам ===
melted = df.melt(id_vars="Class", value_vars=features)

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted, x="variable", y="value", hue="Class")
plt.title("📊 Распределение признаков по классам")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 6. Средние значения признаков по классам ===
means = df.groupby("Class")[features].mean().T
means.columns = [class_names[i] for i in means.columns]
print("📌 Средние значения признаков по классам:\n")
print(means.round(3))
