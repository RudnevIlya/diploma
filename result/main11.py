import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Настройка границ классов ===
class_ranges = {
    0: (2.2, 3.94),  
    1: (3.95, 4.19),  
    2: (4.2, 5.0)  
}

# class_ranges = {
#     0: (4, 4.15),  
#     1: (4.16, 4.3),  
#     2: (4.31, 5.0)  
# }
class_names = ["низкий", "средний", "высокий"]

def map_to_class(r):
    for cls, (low, high) in class_ranges.items():
        if low < r <= high:
            return cls
    return None  # фильтруем всё, что не попадает в диапазон

# === Загрузка и подготовка ===
df = pd.read_csv("merged_results.csv")
df["Class"] = df["Rating"].apply(map_to_class)
df = df[df["Class"].notna()]  # исключаем те, что вне диапазонов

# Показываем размер классов
counts = counts = df["Class"].astype(int).value_counts().sort_index()
print("📊 Количество записей по классам:")
for cls_id, count in counts.items():
    print(f"{class_names[cls_id]} ({cls_id}): {count}")

# Данные
features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features]
y = df["Class"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === CatBoostClassifier ===
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    verbose=0
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# === Метрики ===
print("\n🎯 CatBoostClassifier")
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=class_names, labels=[0, 1, 2], zero_division=0))

# === Матрица ошибок ===
cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — CatBoostClassifier")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("classification_comparison_custom.png")
print("\n📊 Матрица ошибок сохранена в classification_comparison_custom.png")
