import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Загрузка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]
ratings = df["Rating"].dropna()

# === Статистика ===
print("📈 Основная статистика по Rating:")
print(ratings.describe())

# === Гистограмма ===
plt.figure(figsize=(10, 6))
sns.histplot(ratings, bins=20, kde=True, color="skyblue", edgecolor="black")
plt.title("Распределение пользовательских рейтингов")
plt.xlabel("Rating")
plt.ylabel("Количество книг")

# === Границы текущих классов ===
boundaries = [3.9, 4.3]
for b in boundaries:
    plt.axvline(b, color="red", linestyle="--", label=f"Граница {b}")

plt.legend()
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# === Распределение по кастомным классам ===
def map_custom_class(r):
    if r <= 3.9:
        return "низкий"
    elif r <= 4.3:
        return "средний"
    else:
        return "высокий"

df["RatingClass"] = df["Rating"].apply(map_custom_class)
counts = df["RatingClass"].value_counts()

print("\n📊 Распределение по классам (по границам 3.9 / 4.3):")
for cls in ["низкий", "средний", "высокий"]:
    print(f"{cls}: {counts.get(cls, 0)}")
