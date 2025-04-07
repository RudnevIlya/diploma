import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler

# === Загрузка модели и скейлера ===
clf = CatBoostClassifier()
reg = CatBoostRegressor()
clf.load_model("catboost_classifier.cbm")
reg.load_model("catboost_regressor.cbm")
scaler: StandardScaler = joblib.load("scaler.save")

# === Загрузка новых данных ===
df_new = pd.read_csv("new_works.csv")

features = [
    "HD_D", "Average_TFIDF", "Value", "Length",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore"
]

X_new = df_new[features]
X_new_scaled = scaler.transform(X_new)

# === Получение признаков регрессии и объединение ===
reg_preds = reg.predict(X_new_scaled).reshape(-1, 1)
X_new_with_reg = np.hstack([X_new_scaled, reg_preds])

# === Предсказание классов и вероятностей ===
predicted_classes = clf.predict(X_new_with_reg)
predicted_proba = clf.predict_proba(X_new_with_reg)

# === Преобразование классов в оценки (примерно) ===
class_to_rating = {0: 3.7, 1: 4.1, 2: 4.5}
df_new["PredictedClass"] = predicted_classes
df_new["PredictedRating"] = df_new["PredictedClass"].map(class_to_rating)

# === Вывод в консоль всех вероятностей ===
class_labels = ["низкий", "средний", "высокий"]
print("\n📊 Предсказания по произведениям:\n")
for i, row in df_new.iterrows():
    print(f"📘 {row['Title']}:")
    for j, prob in enumerate(predicted_proba[i]):
        print(f"    {class_labels[j]}: {prob:.4f}")
    print(f"    🏷️ Предсказанный рейтинг: {row['PredictedRating']}\n")

# === Сохранение с вероятностями ===
proba_df = pd.DataFrame(predicted_proba, columns=[f"Prob_{label}" for label in class_labels])
df_result = pd.concat([df_new, proba_df], axis=1)
df_result.to_csv("predicted_new_works_full.csv", index=False)
print("✅ Результаты с вероятностями сохранены в 'predicted_new_works_full.csv'")
