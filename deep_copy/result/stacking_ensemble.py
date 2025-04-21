import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

############################################
# Функция загрузки файла с метриками
############################################
def load_metric_file(file_path):
    """
    Загружает CSV-файл с метриками и приводит столбец "Book ID" к строковому типу.
    """
    df = pd.read_csv(file_path)
    if "Book ID" in df.columns:
        df["Book ID"] = df["Book ID"].astype(str).str.strip()
    return df

############################################
# Функция получения общего множества Book IDs
############################################
def get_common_book_ids(dfs):
    """
    Принимает список DataFrame и возвращает пересечение значений столбца "Book ID".
    """
    id_sets = [set(df["Book ID"]) for df in dfs if "Book ID" in df.columns]
    common_ids = set.intersection(*id_sets)
    return sorted(list(common_ids))

############################################
# Функция подготовки данных для одной группы метрик
############################################
def prepare_group_data(df, common_ids):
    """
    Фильтрует DataFrame, оставляя только записи, у которых Book ID содержится в common_ids.
    Сортирует по Book ID, затем отделяет признаки (X) и целевую переменную (y).
    При этом предполагается, что столбец "Rating" – целевая.
    """
    to_remove = [
        "avg_adjacent_similarity", "std_adjacent_similarity", "num_adjacent_pairs",
        "pronoun_ratio", "slang_ratio",
        "metaphor_pair_count", "metaphor_pair_ratio",
        "epithet_density", "tonal_ambiguity_score", "sarcasm_markers", "sentence_depths",
          "Views", "Likes"
    ]
    df = df.drop(columns=[col for col in to_remove if col in df.columns], errors="ignore")
    
    if "overall_sentiment" in df.columns:
        if df["overall_sentiment"].dtype == "object":
            mapping = {"negative": -1, "neutral": 0, "positive": 1, "speech": 0, "skip": 0}
            df["overall_sentiment"] = df["overall_sentiment"].str.lower().map(mapping)

    df_group = df[df["Book ID"].isin(common_ids)].copy()
    df_group.sort_values("Book ID", inplace=True)
    # Если в файле имеются лишние столбцы, оставляем все числовые, кроме Book ID и Rating (Rating в y)
    y = df_group["Rating"].values
    X = df_group.drop(columns=["Book ID", "Rating"], errors="ignore")
    # Заполнить пропуски средними значениями
    X = X.fillna(X.mean())
    return X, y, df_group["Book ID"].values

############################################
# Функция для обучения базовой модели (линейная регрессия)
############################################
def train_basic_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

############################################
# Функция вычисления метрик
############################################
def evaluate_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    kendall_corr, _ = kendalltau(y_true, y_pred)
    ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae,
            "Spearman": spearman_corr, "Kendall": kendall_corr,
            "NDCG": ndcg}

############################################
# Функция визуализации результатов предсказаний
############################################
def plot_predictions(y_true, y_pred, title="Predicted vs Actual Ratings"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(title)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.show()

############################################
# Основной блок: Обучение базовых моделей и стэкинг
############################################
def main():
    # Пути к файлам с разными группами метрик
    lexical_file = "data/lexical_text_metrics.csv"
    semantic_file = "data/semantic_emotional_metrics_all.csv"
    stylistic_file = "data/stylistic_metrics.csv"
    syntactic_file = "data/syntactic_all_metrics.csv"
    
    # Загружаем файлы
    print("Загружаем данные из файлов с метриками...")
    df_lex = load_metric_file(lexical_file)
    df_sem = load_metric_file(semantic_file)
    df_sty = load_metric_file(stylistic_file)
    df_syn = load_metric_file(syntactic_file)
    
    # Находим общее пересечение Book ID
    common_ids = get_common_book_ids([df_lex, df_sem, df_sty, df_syn])
    print("Общее число Book IDs для всех групп:", len(common_ids))
    
    # Подготовим данные для каждой группы
    X_lex, y_lex, ids_lex = prepare_group_data(df_lex, common_ids)
    X_sem, y_sem, ids_sem = prepare_group_data(df_sem, common_ids)
    X_sty, y_sty, ids_sty = prepare_group_data(df_sty, common_ids)
    X_syn, y_syn, ids_syn = prepare_group_data(df_syn, common_ids)
    
    # Проверяем, что y совпадают (целевая переменная должна быть одинаковой)
    # Если рейтинги различаются, можно взять среднее или выполнить другое объединение.
    # Для простоты предполагаем, что y одинаковые во всех группах.
    y_common = y_lex  # используем рейтинги из лексической группы
    n_samples = len(y_common)
    print("Количество образцов:", n_samples)
    
    # Разбиваем данные на обучающую и тестовую выборки (используя индексы, чтобы согласовать все группы)
    indices = np.arange(n_samples)
    from sklearn.model_selection import train_test_split
    idx_train, idx_test, y_train, y_test = train_test_split(indices, y_common, test_size=0.2, random_state=42)
    
    # Для каждой группы формируем обучающие и тестовые матрицы
    X_lex_train, X_lex_test = X_lex.iloc[idx_train].values, X_lex.iloc[idx_test].values
    X_sem_train, X_sem_test = X_sem.iloc[idx_train].values, X_sem.iloc[idx_test].values
    X_sty_train, X_sty_test = X_sty.iloc[idx_train].values, X_sty.iloc[idx_test].values
    X_syn_train, X_syn_test = X_syn.iloc[idx_train].values, X_syn.iloc[idx_test].values
    
    # Обучаем базовые модели для каждой группы
    print("Обучаем базовую модель для лексических метрик...")
    model_lex = train_basic_model(X_lex_train, y_train)
    print("Обучаем базовую модель для семантических метрик...")
    model_sem = train_basic_model(X_sem_train, y_train)
    print("Обучаем базовую модель для стилистических метрик...")
    model_sty = train_basic_model(X_sty_train, y_train)
    print("Обучаем базовую модель для синтаксических метрик...")
    model_syn = train_basic_model(X_syn_train, y_train)
    
    # Получаем предсказания базовых моделей на тестовой выборке
    pred_lex = model_lex.predict(X_lex_test)
    pred_sem = model_sem.predict(X_sem_test)
    pred_sty = model_sty.predict(X_sty_test)
    pred_syn = model_syn.predict(X_syn_test)
    
    # Выводим метрики для каждой модели
    print("\nМетрики для лексических метрик:")
    metrics_lex = evaluate_model_metrics(y_test, pred_lex)
    print("\nМетрики для семантических метрик:")
    metrics_sem = evaluate_model_metrics(y_test, pred_sem)
    print("\nМетрики для стилистических метрик:")
    metrics_sty = evaluate_model_metrics(y_test, pred_sty)
    print("\nМетрики для синтаксических метрик:")
    metrics_syn = evaluate_model_metrics(y_test, pred_syn)
    
    # Подготавливаем DataFrame с предсказаниями базовых моделей по тестовой выборке,
    # используя один и тот же порядок образцов.
    base_preds = pd.DataFrame({
        "pred_lex": pred_lex,
        "pred_sem": pred_sem,
        "pred_sty": pred_sty,
        "pred_syn": pred_syn
    })
    
    # Обучаем мета-модель (stacking) на базовых предсказаниях.
    # Здесь используем простую линейную регрессию.
    meta_model = LinearRegression()
    meta_model.fit(base_preds, y_test)
    ensemble_pred = meta_model.predict(base_preds)
    
    print("\nМетрики для ансамблевой модели (stacking):")
    ensemble_metrics = evaluate_model_metrics(y_test, ensemble_pred)
    
    # Визуализация: scatter plot предсказаний базовой модели и ансамбля.
    plot_predictions(y_test, ensemble_pred, title="Ensemble Predictions vs Actual Ratings")
    
    # Сводная таблица результатов
    results = {
        "Group": ["Lexical", "Semantic", "Stylistic", "Syntactic", "Ensemble"],
        "MSE": [metrics_lex["MSE"], metrics_sem["MSE"], metrics_sty["MSE"], metrics_syn["MSE"], ensemble_metrics["MSE"]],
        "RMSE": [metrics_lex["RMSE"], metrics_sem["RMSE"], metrics_sty["RMSE"], metrics_syn["RMSE"], ensemble_metrics["RMSE"]],
        "MAE": [metrics_lex["MAE"], metrics_sem["MAE"], metrics_sty["MAE"], metrics_syn["MAE"], ensemble_metrics["MAE"]],
        "Spearman ρ": [metrics_lex["Spearman"], metrics_sem["Spearman"], metrics_sty["Spearman"], metrics_syn["Spearman"], ensemble_metrics["Spearman"]],
        "Kendall Tau": [metrics_lex["Kendall"], metrics_sem["Kendall"], metrics_sty["Kendall"], metrics_syn["Kendall"], ensemble_metrics["Kendall"]],
        "NDCG": [metrics_lex["NDCG"], metrics_sem["NDCG"], metrics_sty["NDCG"], metrics_syn["NDCG"], ensemble_metrics["NDCG"]]
    }
    results_df = pd.DataFrame(results)
    print("\nСводная таблица метрик:")
    print(results_df.to_markdown(index=False))
    
    # Сохраняем результаты стэкинга и базовые предсказания
    os.makedirs("models", exist_ok=True)
    results_df.to_csv("models/stacking_results.csv", index=False)
    base_preds.to_csv("models/base_model_predictions.csv", index=False)
    joblib.dump(meta_model, "models/meta_model.pkl")
    print("Результаты и мета-модель сохранены в папке models.")

if __name__ == "__main__":
    main()
