import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv("final_merged_output.csv")

# 🔍 Удаление мусорных значений
df_cleaned = df[df["Rating"] >= 0].copy()

# 🧮 Округление до 0.1 (можно отключить, если не нужно)
df_cleaned["RatingRounded"] = df_cleaned["Rating"].round(1)

# 📊 Вывод статистики
print("📊 Статистика после очистки:")
print(df_cleaned["Rating"].describe())
print("\n🔢 Частоты округлённых значений:")
print(df_cleaned["RatingRounded"].value_counts().sort_index())

# 📈 Построение гистограммы
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["RatingRounded"], bins=20, kde=True, color="mediumseagreen", edgecolor="black")
plt.title("📈 Распределение оценок после очистки (округление до 0.1)", fontsize=14)
plt.xlabel("Оценка (RatingRounded)", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
