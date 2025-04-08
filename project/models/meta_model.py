import os
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_meta_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает финальную мета-модель на основе предсказаний базовых моделей.

    На входе:
      df_train: pd.DataFrame с колонками:
          - ID
          - Pred_Emotion
          - Pred_Lexical
          - Pred_Structure
          - Pred_BERT
          - Rating
      df_test: pd.DataFrame с аналогичной структурой (используется для оценки модели).

    Модель обучается с использованием CatBoostRegressor с параметром l2_leaf_reg=10,
    используя только предсказания базовых моделей в качестве признаков.

    Возвращает:
      pd.DataFrame с колонками:
          - ID
          - PredictedRating

    Дополнительно:
      - Выводит метрики MAE, RMSE, R².
      - Строит scatter plot "Predicted vs Actual" и сохраняет его в 'results/meta_model_plot.png'.
      - Строит bar chart важности признаков и сохраняет его в 'results/meta_model_feature_importance.png'.
      - Сохраняет итоговые предсказания в 'results/final_prediction.csv'.
    """
    # Используем только предсказания базовых моделей как признаки.
    # features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_Structure', 'Pred_BERT']
    features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_BERT']
    X_train = df_train[features]
    y_train = df_train['Rating']
    X_test = df_test[features]
    y_test = df_test['Rating']
    ids_test = df_test['ID']
    
    # Инициализация CatBoostRegressor с регуляризацией l2_leaf_reg=10
    try:
        model = CatBoostRegressor(task_type='GPU', l2_leaf_reg=10, random_seed=42, verbose=0)
        print("Используется GPU для финальной мета-модели.")
    except Exception as e:
        print("GPU недоступен или произошла ошибка, переключение на CPU. Подробности:", e)
        model = CatBoostRegressor(task_type='CPU', l2_leaf_reg=10, random_seed=42, verbose=0)
    
    # Обучение модели
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("Final Meta-Model Metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    
    # Построение scatter plot: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Meta-Model: Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    scatter_path = os.path.join('results', 'meta_model_plot.png')
    plt.savefig(scatter_path)
    plt.show()
    print("Scatter plot сохранён:", scatter_path)
    
    # Построение bar chart важности признаков (если модель поддерживает)
    try:
        importances = model.get_feature_importance()
        plt.figure(figsize=(8, 6))
        plt.bar(features, importances)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance of Meta-Model")
        bar_chart_path = os.path.join('results', 'meta_model_feature_importance.png')
        plt.savefig(bar_chart_path)
        plt.show()
        print("Bar chart важности признаков сохранён:", bar_chart_path)
    except Exception as e:
        print("Не удалось получить важность признаков для финальной мета-модели:", e)
    
    # Итоговый DataFrame с предсказаниями
    result_df = pd.DataFrame({'ID': ids_test, 'PredictedRating': y_pred})
    final_prediction_path = os.path.join('results', 'final_prediction.csv')
    result_df.to_csv(final_prediction_path, index=False)
    print("Итоговые предсказания сохранены в:", final_prediction_path)
    
    return result_df

if __name__ == '__main__':
    # Пример использования:
    # Файл 'meta_model_input.csv' должен содержать колонки:
    # ID, Pred_Emotion, Pred_Lexical, Pred_Structure, Pred_BERT, Rating
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корневая папка проекта
    data_path = os.path.join(base_dir, 'data', 'meta_model_input.csv')
    df = pd.read_csv(data_path)
    
    # Делим данные на train и test (например, 50/50)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    
    predictions_df = train_meta_model(df_train, df_test)
    print(predictions_df)
