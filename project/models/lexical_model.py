import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Пытаемся импортировать GPU-ускоренную версию Ridge из cuML
try:
    from cuml.linear_model import Ridge as RidgeGPU
    import cudf
    use_gpu = True
except ImportError:
    from sklearn.linear_model import Ridge as RidgeCPU
    use_gpu = False

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель RidgeRegression для предсказания рейтинга художественного текста
    на основе лексико-стилистических признаков и возвращает датафрейм с предсказаниями.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм, содержащий признаки:
            - HD_D
            - Average_TFIDF
            - LexicalDiversity
            - LongWordRatio
            - AbstractNounRatio
            - AdjRatio
            - VerbRatio
        а также колонки 'Rating' и 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Вычисляет метрики MAE, RMSE, R².
    - Сохраняет результаты предсказаний в 'results/pred_lexical.csv'.
    """
    # Определяем список признаков
    features = ['HD_D', 'Average_TFIDF', 'LexicalDiversity', 'LongWordRatio',
                'AbstractNounRatio', 'AdjRatio', 'VerbRatio']
    X = df[features]
    y = df['Rating']
    ids = df['ID']
    
    # Разбиваем данные на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )
    
    # Обучение модели с использованием GPU (cuML) или CPU (sklearn)
    if use_gpu:
        # Преобразуем данные в cudf DataFrame/Series
        X_train_cudf = cudf.DataFrame.from_pandas(X_train)
        X_test_cudf = cudf.DataFrame.from_pandas(X_test)
        y_train_cudf = cudf.Series(y_train.values)
        
        model = RidgeGPU()
        model.fit(X_train_cudf, y_train_cudf)
        # Предсказания (результат возвращается в виде cudf Series)
        y_pred = model.predict(X_test_cudf).to_numpy()
    else:
        model = RidgeCPU()
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
    result_df.to_csv(os.path.join('results', 'pred_lexical.csv'), index=False)
    
    return result_df


def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель RidgeRegression для предсказания рейтинга художественного текста
    на основе лексико-стилистических признаков и возвращает датафрейм с предсказаниями.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм, содержащий признаки:
            - HD_D
            - Average_TFIDF
            - LexicalDiversity
            - LongWordRatio
            - AbstractNounRatio
            - AdjRatio
            - VerbRatio
        а также колонки 'Rating' и 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Вычисляет метрики MAE, RMSE, R².
    - Сохраняет результаты предсказаний в 'results/pred_lexical.csv'.
    """
    # Определяем список признаков
    features = ['HD_D', 'Average_TFIDF', 'LexicalDiversity', 'LongWordRatio',
                'AbstractNounRatio', 'AdjRatio', 'VerbRatio']
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
    
    # Обучение модели с использованием GPU (cuML) или CPU (sklearn)
    if use_gpu:
        # Преобразуем данные в cudf DataFrame/Series
        X_train_cudf = cudf.DataFrame.from_pandas(X_train)
        X_test_cudf = cudf.DataFrame.from_pandas(X_test)
        y_train_cudf = cudf.Series(y_train.values)
        
        model = RidgeGPU()
        model.fit(X_train_cudf, y_train_cudf)
        # Предсказания (результат возвращается в виде cudf Series)
        y_pred = model.predict(X_test_cudf).to_numpy()
    else:
        model = RidgeCPU()
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
    result_df.to_csv(os.path.join('results', 'pred_lexical.csv'), index=False)
    
    return result_df



if __name__ == '__main__':
    # Определяем базовую директорию проекта (папка project/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Формируем путь к файлу с данными
    data_path = os.path.join(base_dir, 'data', 'texts_with_features.csv')
    
    # Читаем данные
    df = pd.read_csv(data_path)
    
    predictions_df = train_and_predict(df)
    print(predictions_df)
