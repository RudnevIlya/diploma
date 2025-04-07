import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Загрузка и подготовка данных ===
df = pd.read_csv("final_merged_output_with_bert.csv")
df = df[df["Rating"] >= 1]  # фильтруем мусор

# Классификация по границам
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

# === 2. Добавим новые признаки ===
df["Expressiveness"] = (
    df["ExclamationCount"].fillna(0) +
    df["QuotesCount"].fillna(0)
) * df["Emotional"].fillna(0)

df["LogLength"] = np.log1p(df["Length"].fillna(0))

# === 3. Финальный список признаков ===
features = [
"Emotional",
    "AvgWordLength", "AvgSentenceLength",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","AbstractNounRatio",
    "SimLow", "SimMid", "SimHigh"
]

df = df.dropna(subset=features + ["Class"])
X = df[features].values
y = df["Class"].values

# === 4. Деление на train/test ===
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# === 5. Обучение модели ===
model = LGBMClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# === 6. Метрики ===
print("\n📊 === Модель с Expressiveness + LogLength ===")
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("📊 F1 Macro:", round(f1_score(y_test, y_pred, average='macro'), 4))
print("📊 F1 Weighted:", round(f1_score(y_test, y_pred, average='weighted'), 4))
print("⚖️ Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# === 7. Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предсказано")
plt.ylabel("Истинный класс")
plt.title("Confusion Matrix с новыми признаками")
plt.tight_layout()
plt.show()
