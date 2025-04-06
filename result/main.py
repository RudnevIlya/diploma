import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Загружаем файл
df = pd.read_csv("merged_results.csv")

# Фильтруем строки с корректными рейтингами
df = df[df["Rating"] >= 0]

# Выбираем числовые признаки
feature_columns = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[feature_columns]
y = df["Rating"]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель с использованием GPU
model = XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ R²: {r2:.4f}")
