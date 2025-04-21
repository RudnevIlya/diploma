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
    lexical_file = "data/lexical_text_metrics.csv"
    semantic_file = "data/semantic_emotional_metrics_all.csv"
    stylistic_file = "data/stylistic_metrics.csv"
    syntactic_file = "data/syntactic_all_metrics.csv"
    
    df_lex = pd.read_csv(lexical_file)
    df_sem = pd.read_csv(semantic_file)
    df_sty = pd.read_csv(stylistic_file)
    df_syn = pd.read_csv(syntactic_file)
    
    # Приводим столбец "Book ID" ко строковому типу для всех DataFrame
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
    Предобрабатывает объединённые данные:
      - Удаляет записи без "Rating".
      - Удаляет нежелательные столбцы.
      - Преобразует overall_sentiment (строка) в числовой формат (negative:-1, neutral:0, positive:1, speech/skip:0).
      - Удаляет "Book ID".
      - Отбирает только числовые признаки, заполняет пропуски, масштабирует признаки.
    
    Возвращает: X_scaled, y, scaler и список имен признаков.
    """
    df = df.dropna(subset=["Rating"])
    to_remove = [
        # "avg_adjacent_similarity", "std_adjacent_similarity", "num_adjacent_pairs",
        # "pronoun_ratio", "slang_ratio",
        # "metaphor_pair_count", "metaphor_pair_ratio",
        # "epithet_density", "tonal_ambiguity_score", "sarcasm_markers", "sentence_depths",
        "Views", "Likes"
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
# Функция для логарифмического преобразования целевой переменной
############################################
def transform_target(y, offset=1e-6):
    """
    Преобразует целевую переменную y по формуле:
       y_trans = log(y + offset)
    """
    return np.log(y + offset)

def inverse_transform_target(y_trans, offset=1e-6):
    """
    Инвертирует логарифмическое преобразование:
       y = exp(y_trans) - offset
    """
    return np.exp(y_trans) - offset

############################################
# Обучение модели XGBRegressor с L1-регуляризацией
############################################
def train_xgb_model(X_train, y_train):
    xgb_model = XGBRegressor(random_state=42, tree_method='gpu_hist', n_jobs=-1)
    param_grid = {
        "n_estimators": [200],
        "max_depth": [None],
        "learning_rate": [0.01],
        "reg_alpha": [0.1]
    }
    from sklearn.model_selection import GridSearchCV, KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Лучшие параметры модели XGBRegressor:", grid_search.best_params_)
    return grid_search.best_estimator_

############################################
# Функция оценки дополнительных метрик
############################################
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
# Визуализация
############################################
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Feature Importances from XGBRegressor")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title("Predicted vs Actual Ratings")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.show()

############################################
# Основной блок
############################################
def main():
    df = load_and_merge_metrics()
    print("Общее число записей после объединения:", df.shape)
    
    X, y, scaler, feature_names = preprocess_data(df)
    # print("Размер признакового пространства:", X.shape)
    
    # # Преобразуем целевую переменную логарифмически
    # offset = 1e-6
    # y_trans = transform_target(y, offset)
    
    # X_train, X_test, y_train_trans, y_test_trans = train_test_split(X, y_trans, test_size=0.2, random_state=42)
    # print("Размер обучающей выборки:", X_train.shape)
    # print("Размер тестовой выборки:", X_test.shape)
    
    # model = train_xgb_model(X_train, y_train_trans)
    
    # # Предсказания в лог-пространстве, затем обратное преобразование
    # y_pred_trans = model.predict(X_test)
    # y_pred = inverse_transform_target(y_pred_trans, offset)
    # y_test = inverse_transform_target(y_test_trans, offset)
    
    # # Вычисляем основные метрики, в том числе ранговые и NDCG
    # mse, rmse, mae, spearman_corr, kendall_corr, ndcg = evaluate_model_metrics(y_test, y_pred)
    
    # # Визуализация важности признаков и предсказаний
    # plot_feature_importances(model, feature_names)
    # plot_predictions(y_test, y_pred)
    
    # # Сохраняем модель, scaler и список признаков
    # os.makedirs("models", exist_ok=True)
    # joblib.dump(model, "models/xgb_rating_model.pkl")
    # joblib.dump(scaler, "models/scaler.pkl")
    pd.Series(feature_names, name="features").to_csv("models/feature_names.csv", index=False)
    print("Обученная модель, scaler и список признаков сохранены.")

if __name__ == '__main__':
    main()
