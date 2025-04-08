import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Задаем путь к данным
data_path = os.path.join("data", "texts_with_features.csv")
df = pd.read_csv(data_path)

df = df[df["Rating"] >= 3.5]
df = df[df["Rating"] <= 4.98]

# Выведем список столбцов для ориентира
print("Столбцы в датасете:", df.columns.tolist())

# Исключим нечисловые/текстовые столбцы (Author, Title, AuthorRus), а также, если требуется, признаки для которых нет смысловой нагрузки
cols_to_exclude = ["ID", "Author", "Title", "AuthorRus"]
# Можно оставить остальные признаки, включая HD_D, Average_TFIDF, Emotional, Length, AvgWordLength, AvgSentenceLength,
# LongWordRatio, LexicalDiversity, SentimentScore, VerbRatio, AdjRatio, QuotesCount, NegationCount, ExclamationCount, AbstractNounRatio, SimLow, SimMid, SimHigh
# Если какие-либо из этих признаков содержат пропуски или требуют преобразования – это следует обработать отдельно.
features = ["HD_D"]

# Можно посмотреть статистику по признакам:
print(df[features].describe())

# Разбиваем данные на обучающую и тестовую выборки (например, 80/20)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Формируем X и y
X_train = df_train[features]
y_train = df_train["Rating"]
X_test = df_test[features]
y_test = df_test["Rating"]

# Инициализируем CatBoostRegressor (можно попробовать и другие модели, например, Ridge)
try:
    model = CatBoostRegressor(task_type="GPU", l2_leaf_reg=10, random_seed=42, verbose=0)
    print("Используется GPU для CatBoostRegressor.")
except Exception as e:
    print("GPU не доступен, используется CPU. Подробности:", e)
    model = CatBoostRegressor(random_seed=42, l2_leaf_reg=10, verbose=0)

# Обучаем модель на исходных признаках
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Вычисляем метрики
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Model on all features Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
