import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Пытаемся использовать GPU-ускоренную версию DecisionTreeRegressor
try:
    from cuml.tree import DecisionTreeRegressor as DecisionTreeRegressorGPU
    import cudf
    use_gpu = True
    print("Используется GPU (cuML) для DecisionTreeRegressor.")
except ImportError:
    from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorCPU
    use_gpu = False
    print("GPU не найден. Используется CPU-версия DecisionTreeRegressor.")

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель DecisionTreeRegressor для оценки художественного текста по синтаксическим признакам.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм, содержащий признаки:
            - Length
            - AvgWordLength
            - AvgSentenceLength
            - QuotesCount
        а также колонки 'Rating' и 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Вычисляются метрики MAE, RMSE, R².
    - Сохраняет результаты предсказаний в 'results/pred_structure.csv'.
    """
    # Определяем признаки
    features = ['Length', 'AvgWordLength', 'AvgSentenceLength', 'QuotesCount']
    # X = df[features]
    # y = df['Rating']
    # ids = df['ID']
    
    # # Разбиваем данные на обучающую и тестовую выборки (80/20)
    # X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    #     X, y, ids, test_size=0.2, random_state=42
    # )
    
    # Обучение модели с использованием GPU (если возможно) или CPU
    if use_gpu:
        # Преобразуем pandas DataFrame/Series в cudf DataFrame/Series
        X_train_gpu = cudf.DataFrame.from_pandas(X_train)
        X_test_gpu = cudf.DataFrame.from_pandas(X_test)
        y_train_gpu = cudf.Series(y_train.values)
        
        model = DecisionTreeRegressorGPU()
        model.fit(X_train_gpu, y_train_gpu)
        # Предсказания (результат - cudf Series, преобразуем в numpy)
        y_pred = model.predict(X_test_gpu).to_numpy()
    else:
        model = DecisionTreeRegressorCPU(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    
    # Формируем датафрейм с результатами
    result_df = pd.DataFrame({
        'ID': id_test,
        'PredictedRating': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    })
    
    # Создаем папку results (если не существует) и сохраняем предсказания
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_structure.csv'), index=False)
    
    return result_df

def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель DecisionTreeRegressor для оценки художественного текста по синтаксическим признакам.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм, содержащий признаки:
            - Length
            - AvgWordLength
            - AvgSentenceLength
            - QuotesCount
        а также колонки 'Rating' и 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Вычисляются метрики MAE, RMSE, R².
    - Сохраняет результаты предсказаний в 'results/pred_structure.csv'.
    """
    # Определяем признаки
    features = ['Length', 'AvgWordLength', 'AvgSentenceLength', 'QuotesCount']
    # X = df[features]
    # y = df['Rating']
    # ids = df['ID']
    
    # # Разбиваем данные на обучающую и тестовую выборки (80/20)
    # X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    #     X, y, ids, test_size=0.2, random_state=42
    # )
    
    X_train = df_train[features]
    y_train = df_train['Rating']
    id_train = df_train['ID']
    
    X_test = df_test[features]
    y_test = df_test['Rating']
    id_test = df_test['ID']

    # Обучение модели с использованием GPU (если возможно) или CPU
    if use_gpu:
        # Преобразуем pandas DataFrame/Series в cudf DataFrame/Series
        X_train_gpu = cudf.DataFrame.from_pandas(X_train)
        X_test_gpu = cudf.DataFrame.from_pandas(X_test)
        y_train_gpu = cudf.Series(y_train.values)
        
        model = DecisionTreeRegressorGPU()
        model.fit(X_train_gpu, y_train_gpu)
        # Предсказания (результат - cudf Series, преобразуем в numpy)
        y_pred = model.predict(X_test_gpu).to_numpy()
    else:
        model = DecisionTreeRegressorCPU(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    
    # Формируем датафрейм с результатами
    result_df = pd.DataFrame({
        'ID': id_test,
        'PredictedRating': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    })
    
    # Создаем папку results (если не существует) и сохраняем предсказания
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_structure.csv'), index=False)
    
    return result_df


if __name__ == '__main__':
    # Определяем базовую директорию проекта (папка project/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Путь к файлу с данными
    data_path = os.path.join(base_dir, 'data', 'texts_with_features.csv')
    
    # Читаем данные
    df = pd.read_csv(data_path)
    
    # Обучаем модель и получаем предсказания
    predictions_df = train_and_predict(df)
    print(predictions_df)
