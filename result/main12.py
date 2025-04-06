import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

# === Настройка границ классов ===
class_ranges = {
    0: (2.5, 3.99),   # низкий
    1: (3.99, 4.2),   # средний
    2: (4.2, 5.0)    # высокий
}
class_names = ["низкий", "средний", "высокий"]

class_weights = [1.0, 1.2, 1.0]  # если средний класс слабый

def map_to_class(r):
    for cls, (low, high) in class_ranges.items():
        if low < r <= high:
            return cls
    return None

# === Загрузка и подготовка ===
df = pd.read_csv("merged_results.csv")
df["Class"] = df["Rating"].apply(map_to_class)
df = df[df["Class"].notna()]  # удалим те, что вне диапазона

print("📊 Количество записей по классам:")
for cls_id, count in df["Class"].astype(int).value_counts().sort_index().items():
    print(f"{class_names[cls_id]} ({cls_id}): {count}")

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Class"].astype(int).values

# === Кросс-валидация ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_list = []
f1_macro_list = []
precision_list = []
recall_list = []

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
        verbose=0,
        class_weights=class_weights
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)

    acc_list.append(acc)
    f1_macro_list.append(f1)
    precision_list.append(prec)
    recall_list.append(rec)

    print(f"\n--- Fold {fold}/5 ---")
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

# === Средние метрики ===
print("\n📊 Средние метрики по фолдам:")
print(f"✔️ Accuracy:     {np.mean(acc_list):.4f}")
print(f"✔️ F1 (macro):   {np.mean(f1_macro_list):.4f}")
print(f"✔️ Precision:    {np.mean(precision_list):.4f}")
print(f"✔️ Recall:       {np.mean(recall_list):.4f}")
