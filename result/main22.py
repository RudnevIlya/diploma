import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# === Загрузка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# === Классификация по 3 границам ===
low, high = 3.89, 4.25
def to_class(r): return 0 if r <= low else 1 if r <= high else 2
df["Class"] = df["Rating"].apply(to_class)

# === Вывод распределения классов ===
class_counts = df["Class"].value_counts().sort_index()
class_names = ["низкий", "средний", "высокий"]
print("📊 Количество записей по классам:")
for idx, count in class_counts.items():
    print(f"{class_names[idx]} ({idx}): {count}")

# === Признаки ===
features = ["HD_D", "Average_TFIDF", "Length", "LexicalDiversity", "SentimentScore"]
X = df[features]
y = df["Class"]

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Балансировка с помощью SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === Обучение CatBoost ===
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    verbose=0
)
model.fit(X_resampled, y_resampled)

# === Предсказания ===
y_pred = model.predict(X_test)

# === Метрики ===
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# === Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix: CatBoost + SMOTE")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
plt.savefig("cm_catboost_smote.png")
print("✅ Матрица сохранена в cm_catboost_smote.png")
