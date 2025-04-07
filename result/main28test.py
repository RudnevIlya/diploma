import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# === Загрузка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 1]

# === Границы классов ===
class_ranges = {
    "низкий": (0, 3.9),
    "средний": (3.9, 4.3),
    "высокий": (4.3, 5.01)
}
class_names = list(class_ranges.keys())

def rating_to_class(rating):
    for idx, (label, (low, high)) in enumerate(class_ranges.items()):
        if low <= rating < high:
            return idx
    return len(class_ranges) - 1

# === Признаки ===
features = [
    "Emotional", "AvgSentenceLength", "Average_TFIDF", "HD_D",
    "ExclamationCount", "AbstractNounRatio", "AdjRatio", "VerbRatio"
]

X = df[features]
y_class = df["Rating"].apply(rating_to_class)

# === Масштабирование ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Кривая обучения ===
from sklearn.utils.class_weight import compute_class_weight

# Вычисляем веса вручную
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_class), y=y_class)
clf = CatBoostClassifier(verbose=0, random_state=42, class_weights=class_weights)
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_scaled, y_class,
    cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

# === Построение графика ===
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation")
plt.xlabel("Размер обучающей выборки")
plt.ylabel("Accuracy")
plt.title("Кривая обучения (CatBoostClassifier)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Наивная модель (baseline) ===
major_class = y_class.value_counts().idxmax()
y_dummy = [major_class] * len(y_class)
acc_baseline = accuracy_score(y_class, y_dummy)
print(f"\n🎯 Наивная модель (Baseline Accuracy): {acc_baseline:.3f}")

# === Доверительный интервал ===
splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=42)
accuracies = []
for train_idx, test_idx in splitter.split(X_scaled, y_class):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
ci_low = mean_acc - 1.96 * std_acc / np.sqrt(len(accuracies))
ci_high = mean_acc + 1.96 * std_acc / np.sqrt(len(accuracies))

print(f"📊 Средняя точность: {mean_acc:.3f}")
print(f"📎 95% доверительный интервал: [{ci_low:.3f}, {ci_high:.3f}]")
