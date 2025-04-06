import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# === 1. Настройки ===
USE_GPU = True
DEVICE_ID = '0'  # можно указать '0:1' для мульти-GPU

# === 2. Загрузка и фильтрация ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Параметры GPU и RandomSearch ===
task_type = "GPU" if USE_GPU else "CPU"
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 300, 500],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

base_model = CatBoostRegressor(
    task_type=task_type,
    devices=DEVICE_ID if USE_GPU else None,
    verbose=0,
    random_state=42
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42
)
search.fit(X_train, y_train)

# === 4. Обучение с лучшими параметрами ===
best_model = search.best_estimator_
best_model.fit(X_train, y_train)

# === 5. Оценка модели ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ MAE:", round(mae, 4))
print("✅ R²:", round(r2, 4))
print("\n🎯 Лучшие параметры:")
print(search.best_params_)

# === 6. Важность признаков ===
importances = best_model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n📊 Важность признаков:")
print(importance_df)

# === 7. Визуализация ===
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='orange')
plt.xlabel("Importance")
plt.title("Feature Importance (CatBoost, GPU)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\n📈 График сохранён как feature_importance.png")
