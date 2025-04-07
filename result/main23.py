import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# === Признаки и целевая переменная ===
# features = ['HD_D', 'Average_TFIDF', 'Length', 'LexicalDiversity', 'SentimentScore']
features = [
    "HD_D", "Average_TFIDF", "Value", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio", "LexicalDiversity", "SentimentScore"
]
X = df[features]
y_rating = df['Rating']

# === Классы по границам ===
low, high = 3.89, 4.25
def rating_to_class(r): return 0 if r <= low else 1 if r <= high else 2
y_class = y_rating.apply(rating_to_class)

# === Train/Test Split ===
X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_rating, test_size=0.2, random_state=42, stratify=y_class)

# === Модель классификатора ===
clf = CatBoostClassifier(
    iterations=200, learning_rate=0.03, depth=7,
    l2_leaf_reg=11, random_strength=10, bootstrap_type="MVS",
    # class_weights=[4.0, len(df)/df["Class"].value_counts()[1], 4.0],
    verbose=0
)
clf.fit(X_train, y_train_cls)

# === Модель регрессора ===
reg = CatBoostRegressor(
    iterations=200, learning_rate=0.03, depth=7,
    l2_leaf_reg=11, random_strength=10, bootstrap_type="MVS",
    verbose=0
)
reg.fit(X_train, y_train_reg)

# === Предсказания ===
probs_clf = clf.predict_proba(X_test)
preds_reg = reg.predict(X_test)

# === Преобразование регрессии в вероятности классов ===
def reg_to_probs(values, borders=(low, high)):
    probas = []
    for v in values:
        if v <= borders[0]:
            probas.append([1.0, 0.0, 0.0])
        elif v <= borders[1]:
            probas.append([0.0, 1.0, 0.0])
        else:
            probas.append([0.0, 0.0, 1.0])
    return np.array(probas)

probs_reg = reg_to_probs(preds_reg)

# === Ансамблирование: настрой вес регрессора вручную ===
w_reg = 0.6  # например, 60% регрессия, 40% классификация
w_clf = 1 - w_reg

ensemble_probs = w_reg * probs_reg + w_clf * probs_clf
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# === Метрики ===
print("🎯 [Stacked Ensemble]")
acc = accuracy_score(y_test_cls, ensemble_preds)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test_cls, ensemble_preds, target_names=["низкий", "средний", "высокий"], zero_division=0))

# === Confusion Matrix ===
cm = confusion_matrix(y_test_cls, ensemble_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["низкий", "средний", "высокий"],
            yticklabels=["низкий", "средний", "высокий"])
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Confusion Matrix: Stacked Ensemble")
plt.tight_layout()
plt.savefig("cm_stacked_ensemble.png")
print("✅ Матрица сохранена в cm_stacked_ensemble.png")
