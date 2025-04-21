import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

def load_ratings(csv_file):
    """
    Загружает CSV-файл с метаданными и возвращает массив значений Rating.
    """
    df = pd.read_csv(csv_file)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    return df["Rating"].values

def evaluate_model_metrics(y_true, y_pred):
    """
    Вычисляет и печатает MSE, RMSE, MAE, Spearman ρ, Kendall τ и NDCG.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    kendall_corr, _ = kendalltau(y_true, y_pred)
    ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
    print("MSE:       ", mse)
    print("RMSE:      ", rmse)
    print("MAE:       ", mae)
    print("Spearman ρ:", spearman_corr)
    print("Kendall τ: ", kendall_corr)
    print("NDCG:      ", ndcg)
    return mse, rmse, mae, spearman_corr, kendall_corr, ndcg

def main():
    csv_file = "./data/_merged_filtered.csv"
    print("Загружаем истинные рейтинги...")
    y = load_ratings(csv_file)
    print(f"Всего текстов: {len(y)}")
    
    # Простая модель: всегда предсказываем среднее значение
    y_mean = y.mean()
    print(f"Средний рейтинг (константное предсказание): {y_mean:.6f}")
    y_pred = np.full_like(y, y_mean)
    
    # Вычисляем метрики для константного предсказания
    print("\nМетрики для baseline-модели (constant mean prediction):")
    evaluate_model_metrics(y, y_pred)

if __name__ == "__main__":
    main()
