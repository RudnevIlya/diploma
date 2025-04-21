import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from tqdm import tqdm

# Допустим, у вас есть два набора признаков:
# 1) Расчитанные лингвистические метрики (из файла, например, "all_metrics.csv")
# 2) TF-IDF признаки (из файла "tfidf_features.csv")

def load_all_metrics(file_path="data/all_metrics.csv"):
    df = pd.read_csv(file_path)
    # Приводим Rating к числовому типу
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    return df

def load_tfidf_features(file_path="data/tfidf_features.csv"):
    df = pd.read_csv(file_path)
    return df

def evaluate_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    spearman_corr, _ = spearmanr(y_true, y_pred)
    kendall_corr, _ = kendalltau(y_true, y_pred)
    ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Spearman": spearman_corr,
            "Kendall": kendall_corr, "NDCG": ndcg}

# Функция для обучения базовой модели на определённом наборе признаков
def train_basic_model(X, y):
    # Простая линейная регрессия как базовая модель
    model = LinearRegression()
    model.fit(X, y)
    return model

def stack_models(preds, meta_model=LinearRegression()):
    """
    Принимает матрицу предсказаний от базовых моделей (каждая колонка – предсказания одной модели) 
    и обучает мета-модель для финального предсказания.
    """
    meta_model.fit(preds, y_train)
    return meta_model

# Пример реализации стэкинга:
def stacking_ensemble(X_list, y, cv_splits=5):
    """
    X_list: список матриц признаков для разных моделей (базовых наборов признаков)
    Возвращает финальные предсказания мета-модели, обученной на out-of-fold предсказаниях базовых моделей.
    """
    n_models = len(X_list)
    n_samples = y.shape[0]
    # Будем использовать 5-кратную кросс-валидацию, чтобы получить out-of-fold предсказания
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((n_samples, n_models))
    final_preds = np.zeros(n_samples)
    
    for train_idx, val_idx in kf.split(X_list[0]):
        preds_fold = []
        for i in range(n_models):
            X_train, X_val = X_list[i][train_idx], X_list[i][val_idx]
            y_train_fold = y[train_idx]
            model = train_basic_model(X_train, y_train_fold)
            pred = model.predict(X_val)
            preds_fold.append(pred.reshape(-1,1))
        preds_fold = np.hstack(preds_fold)
        oof_preds[val_idx] = preds_fold
    
    # Теперь обучим мета-модель на oof_preds
    meta_model = LinearRegression()
    meta_model.fit(oof_preds, y)
    
    # Предсказываем для всей выборки (или на отложенной выборке)
    # Здесь в качестве примера возьмем oof_preds как финальные предсказания
    final_preds = meta_model.predict(oof_preds)
    
    metrics = evaluate_model_metrics(y, final_preds)
    print("Ensemble метрики:", metrics)
    return final_preds, meta_model

def main():
    # Загрузка базовых наборов признаков
    df_metrics = load_all_metrics("data/all_metrics.csv")   # рассчитанные метрики
    df_tfidf = load_tfidf_features("data/tfidf_features.csv") # TF-IDF признаки
    
    # Выделяем целевую переменную
    y = df_metrics["Rating"].values
    
    # Из рассчитанных метрик выделяем нужные признаки (например, убираем ненужные столбцы, если они ещё есть)
    features_metrics = df_metrics.drop(columns=["Rating", "Book ID"], errors="ignore").values
    # Из TF-IDF используем признак без Book ID
    features_tfidf = df_tfidf.drop(columns=["Book ID"], errors="ignore").values
    
    # Масштабируем признаки для обоюдной обработки
    scaler1 = StandardScaler()
    X_metrics = scaler1.fit_transform(features_metrics)
    
    scaler2 = StandardScaler(with_mean=False)  # TF-IDF матрица обычно разреженная, поэтому with_mean=False
    X_tfidf = scaler2.fit_transform(features_tfidf)
    
    # Разбиваем данные на обучающую и тестовую выборки (одинаковые индексы для всех наборов)
    from sklearn.model_selection import train_test_split
    idx = np.arange(y.shape[0])
    idx_train, idx_test, y_train, y_test = train_test_split(idx, y, test_size=0.2, random_state=42)
    
    X_metrics_train, X_metrics_test = X_metrics[idx_train], X_metrics[idx_test]
    X_tfidf_train, X_tfidf_test = X_tfidf[idx_train], X_tfidf[idx_test]
    
    # Обучаем отдельные модели
    model_metrics = train_basic_model(X_metrics_train, y_train)
    model_tfidf = train_basic_model(X_tfidf_train, y_train)
    
    pred_metrics = model_metrics.predict(X_metrics_test)
    pred_tfidf = model_tfidf.predict(X_tfidf_test)
    
    # Выводим метрики для каждой отдельной модели
    print("Метрики модели на рассчитанных признаках:")
    metrics1 = evaluate_model_metrics(y_test, pred_metrics)
    print("Метрики модели TF-IDF:")
    metrics2 = evaluate_model_metrics(y_test, pred_tfidf)
    
    # Стэкинг: объединяем предсказания базовых моделей (на обучающих данных, используя кросс-валидацию)
    # Для упрощения здесь построим ансамбль на полностью обученной выборке
    preds_baseline = np.vstack([model_metrics.predict(X_metrics), model_tfidf.predict(X_tfidf)]).T
    meta_model = LinearRegression()
    meta_model.fit(preds_baseline, y)  # обучаем мета-модель на всех данных
    # Предсказываем для тестовой выборки
    preds_test = np.vstack([model_metrics.predict(X_metrics_test), model_tfidf.predict(X_tfidf_test)]).T
    ensemble_preds = meta_model.predict(preds_test)
    
    print("Метрики ансамблевой модели (stacking):")
    ensemble_metrics = evaluate_model_metrics(y_test, ensemble_preds)
    
    # Сохраняем результаты (если необходимо)
    results = {
        "Metrics_Model": metrics1,
        "TFIDF_Model": metrics2,
        "Ensemble": ensemble_metrics
    }
    results_df = pd.DataFrame(results, index=["MSE", "RMSE", "MAE", "Spearman", "Kendall", "NDCG"])
    print("\nСводная таблица метрик:")
    print(results_df)
    
if __name__ == '__main__':
    main()
