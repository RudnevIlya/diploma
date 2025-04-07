import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

# === 1. Загрузка данных ===
df = pd.read_csv("final_merged_output_with_bert.csv")
df = df[df["Rating"] >= 1]

# === 2. Классификация рейтинга ===
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

# === 3. Выбор признаков ===
features = [
"HD_D", "Average_TFIDF", "Emotional",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","AbstractNounRatio"
]
# features = [
# "HD_D", "Average_TFIDF", "Emotional", "Length",
#     "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
#     "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio"
# ]
df = df.dropna(subset=features + ["Class"])

X = df[features].values
y = df["Class"].values

# === 4. Разделение на train/test ===
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

# === 6. Предсказание вероятностей
y_proba = model.predict_proba(X_test)

# === 7. Применение порога для "высокий"
threshold_high = 0.35
y_pred_custom = []

for probs in y_proba:
    if probs[2] >= threshold_high:
        y_pred_custom.append(2)
    else:
        y_pred_custom.append(np.argmax(probs[:2]))

y_pred_custom = np.array(y_pred_custom)

# === 8. Метрики
print(f"\n📊 === Модель с порогом ≥ {threshold_high} для 'высокий' ===")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred_custom))
print("📊 F1 Macro:", f1_score(y_test, y_pred_custom, average='macro'))
print("📊 F1 Weighted:", f1_score(y_test, y_pred_custom, average='weighted'))
print("⚖️ Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_custom))
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred_custom, target_names=class_names, digits=3))

# === 9. Матрица ошибок
cm = confusion_matrix(y_test, y_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (порог {threshold_high} для 'высокий')")
plt.tight_layout()
plt.show()
