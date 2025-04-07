import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# Признаки и целевая переменная
# features = ["HD_D", "Average_TFIDF", "Length", "LexicalDiversity", "SentimentScore"]
features = [
    "HD_D", "Average_TFIDF", "Value", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio", "LexicalDiversity", "SentimentScore"
]
X = df[features]
low, high = 3.89, 4.25
def rating_to_class(r): return 0 if r <= low else 1 if r <= high else 2
# y_class = y_rating.apply(rating_to_class)
y_binary = df["Rating"].apply(lambda r: rating_to_class(r))  # 1 - высокий, 0 - не высокий

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Балансировка классов
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y_binary)

# Тренировочный/тестовый сплит
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.15, random_state=42
)

# === Регрессор (вспомогательный) ===
reg = CatBoostRegressor(verbose=0)
reg.fit(X_train, y_train)

# Добавляем предсказания регрессора в качестве нового признака
X_train_reg = np.hstack([X_train, reg.predict(X_train).reshape(-1, 1)])
X_test_reg = np.hstack([X_test, reg.predict(X_test).reshape(-1, 1)])

# === Классификатор с дополнительным признаком ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_reg, y_train)
y_pred = clf.predict(X_test_reg)

# === Метрики ===
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=["не высокий", "средний", "высокий"]))

# === Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["не высокий", "средний", "высокий"],
            yticklabels=["не высокий", "высокий"], cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (High vs Rest with Regression Feature)")
plt.tight_layout()
plt.savefig("confusion_high_vs_rest_with_reg.png")
print("✅ Матрица ошибок сохранена в 'confusion_high_vs_rest_with_reg.png'")
