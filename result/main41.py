import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

# === Загрузка и подготовка ===
df = pd.read_csv("final_merged_output_with_bert.csv")
df = df[df["Rating"] >= 1]

# === Бинарная цель: 1 — высокий рейтинг, 0 — остальное
df["Target"] = (df["Rating"] >= 4.25).astype(int)

features = [
    "Emotional", "AvgSentenceLength", "Average_TFIDF", "HD_D",
    "ExclamationCount", "AbstractNounRatio", "AdjRatio", "VerbRatio",
    "SimLow", "SimMid", "SimHigh"
]

df = df.dropna(subset=features + ["Target"])
X = df[features].values
y = df["Target"]

# === Разбиение
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# === Балансировка
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# === Обучение
model = LGBMClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

# === Метрики
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n📊 === Бинарная классификация: Высокий рейтинг ===")
print(f"🎯 Accuracy:       {acc:.4f}")
print(f"📎 F1 Score:       {f1:.4f}")
print(f"📌 Precision:      {precision:.4f}")
print(f"📈 Recall:         {recall:.4f}")
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["не высокий", "высокий"]))

# === Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["не высокий", "высокий"])
disp.plot(cmap="Greens", values_format="d")
plt.title("Confusion Matrix — Бинарная классификация")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# === Предсказания вероятностей
y_proba = model.predict_proba(X_test)[:, 1]  # вероятность "высокий"

# === Массив порогов
thresholds = np.linspace(0.0, 1.0, 101)

precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    f1s.append(f1_score(y_test, y_pred_thresh, zero_division=0))

# === Найдём лучший f1
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

print(f"⭐ Лучший F1 = {f1s[best_idx]:.3f} при threshold = {best_threshold:.2f}")
print(f"🔎 Precision = {precisions[best_idx]:.3f}, Recall = {recalls[best_idx]:.3f}")

# === Визуализация
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label="Precision", linestyle='--')
plt.plot(thresholds, recalls, label="Recall", linestyle='-.')
plt.plot(thresholds, f1s, label="F1 Score", linewidth=2)

plt.axvline(best_threshold, color='gray', linestyle=':', label=f"Best Threshold = {best_threshold:.2f}")
plt.xlabel("Порог (threshold)")
plt.ylabel("Метрика")
plt.title("Зависимость Precision / Recall / F1 от порога")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

threshold = 0.75  # или другой, исходя из графика
y_proba = model.predict_proba(X_test)[:, 1]

# Получаем предсказания по новому порогу
y_pred_custom = (y_proba >= threshold).astype(int)

# Метрики
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

print(f"\n⚙️ Порог вероятности: {threshold}")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred_custom))
print("📈 Recall:", recall_score(y_test, y_pred_custom))
print("📌 Precision:", precision_score(y_test, y_pred_custom))
print("📎 F1 Score:", f1_score(y_test, y_pred_custom))

print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred_custom, digits=3))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === y_test — настоящие метки (мультикласс)
# === y_proba — вероятности, полученные от model.predict_proba(X_test)
# shape: (n_samples, 3)

threshold_high = 0.75  # порог для класса "высокий" (индекс 2)

# === Кастомные предсказания с приоритетом "высокий"
y_pred_custom = []

for probs in y_proba:
    if probs[2] >= threshold_high:
        y_pred_custom.append(2)  # "высокий"
    else:
        y_pred_custom.append(np.argmax(probs[:2]))  # выбираем между "низкий" (0) и "средний" (1)

y_pred_custom = np.array(y_pred_custom)

# === Метрики
print(f"\n📊 === Трёхклассовая модель с порогом для 'высокий' (≥ {threshold_high}) ===")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred_custom))
print("📊 F1 Macro:", f1_score(y_test, y_pred_custom, average="macro"))
print("📊 F1 Weighted:", f1_score(y_test, y_pred_custom, average="weighted"))
print("⚖️ Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_custom))
print("\n📝 Classification Report:")
print(classification_report(
    y_test, y_pred_custom,
    target_names=["низкий", "средний", "высокий"],
    digits=3
))

# === Матрица ошибок
cm = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["низкий", "средний", "высокий"], yticklabels=["низкий", "средний", "высокий"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (с порогом для 'высокий')")
plt.tight_layout()
plt.show()
