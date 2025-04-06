import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка и фильтрация данных
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features]
y = df["Rating"]

# Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Параметры для подбора
param_dist = {
    "iterations": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [3, 4, 5, 6, 7],
    "l2_leaf_reg": [1, 3, 5, 7, 9, 11],
    "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
    "random_strength": [1, 2, 3, 5, 10],
}

# Модель
cat_model = CatBoostRegressor(
    verbose=0,
    random_state=42,
    task_type="GPU"  # Убери, если у тебя только CPU
)

# RandomizedSearch
search = RandomizedSearchCV(
    cat_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=2  # или n_jobs=2
)

search.fit(X_train, y_train)

# Лучшая модель
best_model = search.best_estimator_
preds = best_model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

# Вывод
print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R²: {r2:.4f}")
print("\n🎯 Лучшие параметры:")
print(search.best_params_)
