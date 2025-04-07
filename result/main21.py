import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from catboost import CatBoostClassifier, CatBoostRegressor

# === Настройка классов ===
class_ranges = {
    0: (0.0, 3.9),
    1: (3.9, 4.3),
    2: (4.3, 5.0)
}
class_names = ["низкий", "средний", "высокий"]
def to_class(value):
    for cls, (low, high) in class_ranges.items():
        if low <= value <= high:
            return cls
    return None

# === Загрузка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0].copy()
df["Class"] = df["Rating"].apply(to_class)

features = ['HD_D', 'Average_TFIDF', 'Length', 'LexicalDiversity', 'SentimentScore']
X = df[features]
y_class = df["Class"]
y_reg = df["Rating"]

X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# === Классификатор ===
clf = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=6,
    class_weights=[4.0, len(df)/df["Class"].value_counts()[1], 4.0],
    verbose=0
)
clf.fit(X_train, y_train_cls)
clf_preds = clf.predict(X_test)

# === Регрессор ===
reg = CatBoostRegressor(
    iterations=300,
    learning_rate=0.03,
    depth=6,
    verbose=0
)
reg.fit(X_train, y_train_reg)
reg_preds_raw = reg.predict(X_test)
reg_preds_class = np.array([to_class(val) for val in reg_preds_raw])

# === Метрики ===
print("🎯 [Regressor → Классы]")
print(f"Accuracy: {accuracy_score(y_test_cls, reg_preds_class):.4f}")
print(classification_report(y_test_cls, reg_preds_class, target_names=class_names, zero_division=0))

print("\n🎯 [Classifier]")
print(f"Accuracy: {accuracy_score(y_test_cls, clf_preds):.4f}")
print(classification_report(y_test_cls, clf_preds, target_names=class_names, zero_division=0))

# === Confusion Matrix ===
def plot_confusion(true, pred, title, filename):
    cm = confusion_matrix(true, pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 Confusion matrix сохранена в {filename}")

plot_confusion(y_test_cls, reg_preds_class, "Confusion Matrix: Regressor → Классы", "confusion_regressor.png")
plot_confusion(y_test_cls, clf_preds, "Confusion Matrix: Classifier", "confusion_classifier.png")
