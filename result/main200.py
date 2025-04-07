import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier

# Загрузка данных
df = pd.read_csv("merged_results_with_sentiment.csv")
df = df[df["Rating"] >= 0]

# Классы по границам
low, high = 3.89, 4.25
def to_class(r): return 0 if r <= low else 1 if r <= high else 2
df["Class"] = df["Rating"].apply(to_class)

# Признаки
features = [
    "HD_D", "Average_TFIDF", "Value", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio", "LexicalDiversity", "SentimentScore"
]
X = df[features]
y = df["Class"]

# Инициализация модели
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    class_weights=[4.0, len(df)/df["Class"].value_counts()[1], 4.0],
    verbose=0
)

# Отбор признаков
model.fit(X, y)
selector = SelectFromModel(model, prefit=True, threshold="mean")
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()].tolist()

# Повторная модель для оценки
model_cv = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    class_weights=[4.0, len(df)/df["Class"].value_counts()[1], 4.0],
    verbose=0
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(model_cv, X_selected, y, cv=cv, scoring=make_scorer(accuracy_score))
f1_scores = cross_val_score(model_cv, X_selected, y, cv=cv, scoring=make_scorer(f1_score, average='macro'))
prec_scores = cross_val_score(model_cv, X_selected, y, cv=cv, scoring=make_scorer(precision_score, average='macro'))
rec_scores = cross_val_score(model_cv, X_selected, y, cv=cv, scoring=make_scorer(recall_score, average='macro'))

# Вывод результатов
print("📌 Используемые признаки:", selected_features)
print(f"✅ Accuracy:  {np.mean(acc_scores):.4f}")
print(f"✅ F1 Macro:  {np.mean(f1_scores):.4f}")
print(f"✅ Precision: {np.mean(prec_scores):.4f}")
print(f"✅ Recall:    {np.mean(rec_scores):.4f}")
