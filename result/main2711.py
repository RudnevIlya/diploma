import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Конфигурация ===

# Путь к файлам
scaler_path = "scaler.save"
regressor_path = "catboost_regressor.cbm"
classifier_path = "catboost_classifier.cbm"

# Границы классов — значения (min, max) для каждой категории
class_ranges = {
    "оч низкий": (0.0, 3.59),
    "низкий": (3.6, 3.89),
    "средний": (3.89, 4.15),
    "высокий": (4.16, 4.44),
    "оч высокий": (4.45, 5.01)
}
class_names = list(class_ranges.keys())

def rating_to_class(rating):
    for idx, (label, (low, high)) in enumerate(class_ranges.items()):
        if low <= rating < high:
            return idx
    return len(class_ranges) - 1  # fallback (вдруг рейтинг == 5.0)

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# features = [
#     "HD_D", "Average_TFIDF", "Emotional", "Length",
#     "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
#     "LexicalDiversity", "SentimentScore"
# ]

features = [
    "HD_D", "Average_TFIDF", "Emotional", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio"
]

X = df[features]
y_class = df["Rating"].apply(rating_to_class)

# === Стандартизация ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, scaler_path)

# === Балансировка (при необходимости)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y_class)
# X_resampled, y_resampled = X_scaled, y_class

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# === Регрессор (вспомогательный) ===
reg = CatBoostRegressor(verbose=0)
if os.path.exists(regressor_path):
    reg.load_model(regressor_path)
    print("📥 Regressor загружен из файла")
else:
    reg.fit(X_train, y_train)
    # reg.save_model(regressor_path)
    print("💾 Regressor обучен и сохранён")

X_train_ext = np.hstack([X_train, reg.predict(X_train).reshape(-1, 1)])
X_test_ext = np.hstack([X_test, reg.predict(X_test).reshape(-1, 1)])

# === Классификатор ===
clf = CatBoostClassifier(verbose=0)
if os.path.exists(classifier_path):
    clf.load_model(classifier_path)
    print("📥 Classifier загружен из файла")
else:
    clf.fit(X_train_ext, y_train)
    # clf.save_model(classifier_path)
    print("💾 Classifier обучен и сохранён")

# === Метрики ===
y_pred = clf.predict(X_test_ext)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("📊 Classification report:")
print(classification_report(
    y_test, y_pred,
    target_names=class_names
))

# === Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
            yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (CatBoostClassifier with Regression Feature)")
plt.tight_layout()
plt.savefig("confusion_catboost_with_reg.png")
print("✅ Матрица ошибок сохранена в 'confusion_catboost_with_reg.png'")
