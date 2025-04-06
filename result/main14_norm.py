import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

# === Границы классов по квартилям ===
low_thresh = 3.89
high_thresh = 4.25
class_names = ["низкий", "средний", "высокий"]

def map_quantile_class(r):
    if r <= low_thresh:
        return 0
    elif r <= high_thresh:
        return 1
    else:
        return 2

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]
df["Class"] = df["Rating"].apply(map_quantile_class)

print("📊 Количество записей по классам:")
class_counts = df["Class"].astype(int).value_counts().sort_index()
for cls_id, count in class_counts.items():
    print(f"{class_names[cls_id]} ({cls_id}): {count}")

# === Расчёт class_weights (обратная частота) ===
total = len(df)
weights = {cls: total / count for cls, count in class_counts.items()}
class_weights = [weights.get(i, 1.0) for i in range(3)]
print(f"\n⚖️ Class Weights: {class_weights}")

# === Данные ===
features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Class"].astype(int).values

# === Кросс-валидация ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list, prec_list, rec_list = [], [], [], []
best_fold_idx = -1
best_accuracy = -1
best_cm = None
best_labels = None

print("\n🚀 Кросс-валидация (5 фолдов):")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=11,
        random_strength=10,
        bootstrap_type="MVS",
        class_weights=class_weights,
        verbose=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)

    acc_list.append(acc)
    f1_list.append(f1)
    prec_list.append(prec)
    rec_list.append(rec)

    print(f"\n--- Fold {fold}/5 ---")
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    if acc > best_accuracy:
        best_accuracy = acc
        best_fold_idx = fold
        best_cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        best_labels = y_test

# === Средние метрики ===
print("\n📊 Средние метрики по фолдам:")
print(f"✔️ Accuracy:     {np.mean(acc_list):.4f}")
print(f"✔️ F1 (macro):   {np.mean(f1_list):.4f}")
print(f"✔️ Precision:    {np.mean(prec_list):.4f}")
print(f"✔️ Recall:       {np.mean(rec_list):.4f}")

# === Лучшая confusion matrix ===
if best_cm is not None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Предсказано")
    plt.ylabel("Истинно")
    plt.title(f"Матрица ошибок (лучший фолд #{best_fold_idx})")
    plt.tight_layout()
    plt.savefig("best_confusion_matrix.png")
    print(f"\n📊 Матрица ошибок сохранена: best_confusion_matrix.png")
