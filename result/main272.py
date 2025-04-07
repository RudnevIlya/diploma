import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# === Конфигурация ===
class_ranges = {
    "низкий": (0, 3.9),
    "средний": (3.9, 4.3),
    "высокий": (4.3, 5.01),
}
class_names = list(class_ranges.keys())

def rating_to_class(rating):
    for idx, (label, (low, high)) in enumerate(class_ranges.items()):
        if low <= rating < high:
            return idx
    return len(class_ranges) - 1

# === Загрузка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

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

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.15, random_state=42, stratify=y_class
)

# === Oversampling ===
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# === Подбор автоматических весов ===
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_resampled),
    y=y_train_resampled
).tolist()

print("⚖️ class_weights:", weights)

# === Обучение регрессора ===
reg = CatBoostRegressor(verbose=0, random_state=42)
reg.fit(X_train_resampled, y_train_resampled)

# === Обучение классификаторов ===

## 1. С регрессором как фичей
X_train_with_reg = np.hstack([X_train_resampled, reg.predict(X_train_resampled).reshape(-1, 1)])
X_test_with_reg = np.hstack([X_test, reg.predict(X_test).reshape(-1, 1)])

clf_with_reg = CatBoostClassifier(verbose=0, class_weights=weights, random_state=42)
clf_with_reg.fit(X_train_with_reg, y_train_resampled)
y_pred_reg = clf_with_reg.predict(X_test_with_reg)

## 2. Без регрессора
clf_plain = CatBoostClassifier(verbose=0, class_weights=weights, random_state=42)
clf_plain.fit(X_train_resampled, y_train_resampled)
y_pred_plain = clf_plain.predict(X_test)

# === Метрики ===
print("\n📊 === Модель с регрессором ===")
print("Accuracy:", accuracy_score(y_test, y_pred_reg))
print(classification_report(y_test, y_pred_reg, target_names=class_names, digits=3, zero_division=0))

print("\n📊 === Модель без регрессора ===")
print("Accuracy:", accuracy_score(y_test, y_pred_plain))
print(classification_report(y_test, y_pred_plain, target_names=class_names, digits=3, zero_division=0))

# === t-SNE визуализация ===
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame({
    "TSNE-1": X_tsne[:, 0],
    "TSNE-2": X_tsne[:, 1],
    "Class": y_class.map({0: "низкий", 1: "средний", 2: "высокий"})
})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x="TSNE-1", y="TSNE-2", hue="Class", palette="deep", s=60)
plt.title("t-SNE визуализация классов")
plt.tight_layout()
plt.savefig("tsne_classes.png")
plt.show()

importances = clf_plain.get_feature_importance()
for name, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")
