import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Подготовка ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] > 3.0]  # исключаем "низкий" класс

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features]
y_reg = df["Rating"]

# Категоризация рейтинга в 2 класса
def to_class_2(r):
    return 0 if r <= 4.0 else 1  # 0 = средний, 1 = высокий

y_class = y_reg.apply(to_class_2)
class_names = ["средний", "высокий"]

# Train/test split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

# === CatBoostRegressor ===
reg_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    verbose=0
)
reg_model.fit(X_train, y_train_reg)
reg_preds = reg_model.predict(X_test)

# Преобразуем в классы
reg_preds_class = [to_class_2(r) for r in reg_preds]

# === CatBoostClassifier ===
cls_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    verbose=0
)
cls_model.fit(X_train, y_train_cls)
cls_preds = cls_model.predict(X_test)

# === Сравнение ===
print("🎯 [Regressor → Классы]")
print(f"Accuracy: {accuracy_score(y_test_cls, reg_preds_class):.4f}")
print(classification_report(y_test_cls, reg_preds_class, target_names=class_names, labels=[0, 1], zero_division=0))

print("\n🎯 [Classifier]")
print(f"Accuracy: {accuracy_score(y_test_cls, cls_preds):.4f}")
print(classification_report(y_test_cls, cls_preds, target_names=class_names, labels=[0, 1], zero_division=0))

# === Матрицы ошибок ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm1 = confusion_matrix(y_test_cls, reg_preds_class, labels=[0, 1])
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Regressor → Классы")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

cm2 = confusion_matrix(y_test_cls, cls_preds, labels=[0, 1])
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Classifier")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.savefig("classification_comparison_2class.png")
print("\n📊 Матрицы ошибок сохранены в classification_comparison_2class.png")
