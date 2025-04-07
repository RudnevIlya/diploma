import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# === Загрузка данных ===
df = pd.read_csv("final_merged_output_with_bert.csv")
df = df[df["Rating"] >= 1]

# === Границы классов ===
# class_ranges = {
#     "низкий":   (0, 3.85),
#     "средний":  (3.85, 4.25),
#     "высокий":  (4.25, 5.01)
# }

class_ranges = {
    "низкий":   (0, 3.85),
    "средний":  (3.85, 4.25),
    "высокий":  (4.25, 5.01)
}

class_names = list(class_ranges.keys())

def rating_to_class(rating):
    for idx, (label, (low, high)) in enumerate(class_ranges.items()):
        if low <= rating < high:
            return idx
    return len(class_ranges) - 1

# === Признаки ===
manual_features = [
    "HD_D", "Average_TFIDF", "Emotional", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio"
]
cosine_features = ["SimLow", "SimMid", "SimHigh"]
features = manual_features + cosine_features

# features = [
# "Emotional",
#     "AvgWordLength", "AvgSentenceLength",
#     "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","AbstractNounRatio",
#     "SimLow", "SimMid", "SimHigh"
    
# ]

# === Подготовка данных ===
df = df.dropna(subset=features + ["Rating"])
X = df[features]
y = df["Rating"].apply(rating_to_class)

# === Масштабирование ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Разбиение ===
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# === Веса классов ===
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
print(class_weights)

# === Обучение ===
clf = CatBoostClassifier(verbose=0, class_weights=class_weights, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === Метрики ===
print("📊 === Модель: Ручные + Косинусные BERT-признаки ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))

# === Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Manual + Cosine BERT)")
plt.tight_layout()
plt.show()

# === Важность признаков ===
importances = clf.get_feature_importance()
print("\n📌 Важность признаков:")
for name, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")

from sklearn.metrics import f1_score, balanced_accuracy_score

# === Стандартная accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc:.4f}")

# === F1-метрики
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"📊 F1 Macro:     {f1_macro:.4f}")
print(f"📊 F1 Weighted:  {f1_weighted:.4f}")

# === Balanced Accuracy
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"⚖️ Balanced Accuracy: {bal_acc:.4f}")
