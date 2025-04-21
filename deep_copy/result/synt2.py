import os
import math
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

############################################
# Функция объединения файлов с метриками
############################################
def load_and_merge_metrics():
    """
    Загружает файлы с уже рассчитанными метриками и объединяет их по столбцу "Book ID".
    Все значения в столбце "Book ID" приводятся к строковому типу для корректного объединения.
    Из остальных наборов удаляются столбцы, присутствующие в первом, чтобы избежать дублирования.
    """
    lexical_file = "data/lexical_text_metrics.csv"
    semantic_file = "data/semantic_emotional_metrics_all.csv"
    stylistic_file = "data/stylistic_metrics.csv"
    syntactic_file = "data/syntactic_all_metrics.csv"
    
    df_lex = pd.read_csv(lexical_file)
    df_sem = pd.read_csv(semantic_file)
    df_sty = pd.read_csv(stylistic_file)
    df_syn = pd.read_csv(syntactic_file)
    
    # Приводим "Book ID" к строковому типу во всех DataFrame
    for df in [df_lex, df_sem, df_sty, df_syn]:
        if "Book ID" in df.columns:
            df["Book ID"] = df["Book ID"].astype(str).str.strip()
    
    base_cols = set(df_lex.columns) - {"Book ID"}
    df_sem = df_sem.drop(columns=[col for col in df_sem.columns if col in base_cols], errors="ignore")
    df_sty = df_sty.drop(columns=[col for col in df_sty.columns if col in base_cols], errors="ignore")
    df_syn = df_syn.drop(columns=[col for col in df_syn.columns if col in base_cols], errors="ignore")
    
    df = df_lex.merge(df_sem, on="Book ID", how="outer") \
               .merge(df_sty, on="Book ID", how="outer") \
               .merge(df_syn, on="Book ID", how="outer")
    return df

############################################
# Предобработка данных
############################################
def preprocess_data(df):
    """
    Выполняет предобработку объединённых данных:
      - Отбрасывает записи без целевой переменной "Rating".
      - Удаляет нежелательные столбцы:
           "avg_adjacent_similarity", "std_adjacent_similarity", "num_adjacent_pairs",
           "pronoun_ratio", "slang_ratio",
           "metaphor_pair_count", "metaphor_pair_ratio",
           "epithet_density", "tonal_ambiguity_score", "sarcasm_markers", "sentence_depths",
           "Views", "Likes"
      - Преобразует overall_sentiment (если строковый) в числовой формат:
           "negative": -1, "neutral": 0, "positive": 1, "speech"/"skip": 0.
      - Удаляет "Book ID".
      - Выбирает только числовые признаки, заполняет пропуски и масштабирует их.
    
    Возвращает: X_scaled, y, scaler и список имен признаков.
    """
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    df = df.dropna(subset=["Rating"])
    
    to_remove = [
        "avg_adjacent_similarity", "std_adjacent_similarity", "num_adjacent_pairs",
        # "pronoun_ratio", "slang_ratio",
        "metaphor_pair_count", "metaphor_pair_ratio",
        "epithet_density", "tonal_ambiguity_score", "sarcasm_markers", "sentence_depths",
         "View","Size","Likes"
    ]
    df = df.drop(columns=[col for col in to_remove if col in df.columns], errors="ignore")
    
    if df["overall_sentiment"].dtype == "object":
        mapping = {"negative": -1, "neutral": 0, "positive": 1, "speech": 0, "skip": 0}
        df["overall_sentiment"] = df["overall_sentiment"].str.lower().map(mapping)
    
    if "Book ID" in df.columns:
        df = df.drop(columns=["Book ID"])
    
    y = df["Rating"].values
    features = df.drop(columns=["Rating"])
    numeric_features = features.select_dtypes(include=[np.number])
    numeric_features = numeric_features.fillna(numeric_features.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)
    
    return X_scaled, y, scaler, numeric_features.columns.tolist()

############################################
# Обучение модели XGBRegressor с L1-регуляризацией (reg_alpha)
############################################
def train_xgb_model(X_train, y_train):
    xgb_model = XGBRegressor(random_state=42, tree_method='gpu_hist', n_jobs=-1)
    param_grid = {
        "n_estimators": [300],
        "max_depth": [None],
        "learning_rate": [0.01],
        "reg_alpha": [0.1]  # Добавляем L1-регуляризацию
    }
    
    from sklearn.model_selection import GridSearchCV, KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Лучшие параметры модели XGBRegressor:", grid_search.best_params_)
    return grid_search.best_estimator_

############################################
# Визуализация
############################################
def plot_feature_importances(model, feature_names):
    # Извлекаем важности признаков из обученной модели
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
#     plt.title("Feature Importances from XGBRegressor")
#     plt.xlabel("Importance")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     plt.show()

# def plot_predictions(y_test, y_pred):
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Линия идеального соответствия
#     plt.title("Predicted vs Actual Ratings")
#     plt.xlabel("Actual Rating")
#     plt.ylabel("Predicted Rating")
#     plt.tight_layout()
#     plt.show()

def evaluate_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    kendall_corr, _ = kendalltau(y_true, y_pred)
    # Для NDCG требуется 2D массив; воспринимаем тестовый набор как одну группу документов:
    ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Spearman correlation ρ:", spearman_corr)
    print("Kendall Tau:", kendall_corr)
    print("NDCG:", ndcg)
    return mse, rmse, mae, spearman_corr, kendall_corr, ndcg


############################################
# Основной блок
############################################
def main():
    # Объединяем данные
    df = load_and_merge_metrics()
    print("Общее число записей после объединения:", df.shape)
    
    # Предобработка
    X, y, scaler, feature_names = preprocess_data(df)
    print("Размер признакового пространства:", X.shape)
    
    # Разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Размер обучающей выборки:", X_train.shape)
    print("Размер тестовой выборки:", X_test.shape)
    
    # Обучаем модель XGBRegressor с оптимизированными гиперпараметрами
    model = train_xgb_model(X_train, y_train)
    
    # Оцениваем модель
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("XGBRegressor результаты:")
    print("  MSE:", mse)
    print("  MAE:", mae)
    print("  R² :", r2)

    mse, rmse, mae, spearman_corr, kendall_corr, ndcg = evaluate_model_metrics(y_test, y_pred)
    
    # Визуализация важности признаков и предсказаний
    plot_feature_importances(model, feature_names)
    plot_predictions(y_test, y_pred)
    
    # Сохранение модели и предобработчика
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_rating_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    pd.Series(feature_names, name="features").to_csv("models/feature_names.csv", index=False)
    print("Обученная модель, scaler и список признаков сохранены.")

if __name__ == '__main__':
    main()
