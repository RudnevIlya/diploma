import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# Классы: 1 – высокий, 0 – не высокий
df["BinaryClass"] = df["Rating"].apply(lambda r: 1 if r > 4.25 else 0)

# Признаки
# features = ["HD_D", "Average_TFIDF", "Length", "LexicalDiversity", "SentimentScore"]
features = [
    "HD_D", "Average_TFIDF", "Value", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio", "LexicalDiversity", "SentimentScore"
]
X = df[features]
y = df["BinaryClass"]

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Балансировка классов
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42, stratify=y_resampled)

# Обучение модели
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=6,
    verbose=0
)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=["не высокий", "высокий"]))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["не высокий", "высокий"], yticklabels=["не высокий", "высокий"])
plt.xlabel("Предсказано")
plt.ylabel("Истинное значение")
plt.title("Confusion Matrix: высокий vs остальные (RandomOverSampler)")
plt.tight_layout()
plt.savefig("confusion_high_vs_rest_ros.png")
print("✅ Матрица ошибок сохранена в 'confusion_high_vs_rest_ros.png'")
