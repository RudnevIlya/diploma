import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция обучает модель RandomForestRegressor для предсказания рейтинга художественного текста 
    по эмоциональным признакам и возвращает датафрейм с предсказаниями.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм с признаками 'Emotional', 'SentimentScore', 'ExclamationCount', 'NegationCount',
        целевой переменной 'Rating' и идентификатором 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Выводит на экран метрики MAE, RMSE, R².
    - Сохраняет предсказания в 'results/pred_emo.csv'.
    """
    # Определяем список признаков
    feature_cols = ['Emotional', 'SentimentScore', 'ExclamationCount', 'NegationCount']
    X = df[feature_cols]
    y = df['Rating']
    ids = df['ID']
    
    # Разбиваем данные на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )
    
    # Создаем и обучаем модель
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Делаем предсказания на тестовой выборке
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
        'PredictedRating': y_pred
    })
    
    # Создаем папку results, если ее нет, и сохраняем результаты в CSV
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_emo.csv'), index=False)
    
    return result_df

def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Функция обучает модель RandomForestRegressor для предсказания рейтинга художественного текста 
    по эмоциональным признакам и возвращает датафрейм с предсказаниями.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм с признаками 'Emotional', 'SentimentScore', 'ExclamationCount', 'NegationCount',
        целевой переменной 'Rating' и идентификатором 'ID'.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками: 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Выводит на экран метрики MAE, RMSE, R².
    - Сохраняет предсказания в 'results/pred_emo.csv'.
    """
    # Определяем список признаков
    feature_cols = ['Emotional', 'SentimentScore', 'ExclamationCount', 'NegationCount']
    # X = df[feature_cols]
    # y = df['Rating']
    # ids = df['ID']
    
    # # Разбиваем данные на обучающую и тестовую выборки (80/20)
    # X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    #     X, y, ids, test_size=0.2, random_state=42
    # )

    X_train = df_train[feature_cols]
    y_train = df_train['Rating']
    id_train = df_train['ID']
    
    X_test = df_test[feature_cols]
    y_test = df_test['Rating']
    id_test = df_test['ID']
    
    # Создаем и обучаем модель
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Делаем предсказания на тестовой выборке
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
        'PredictedRating': y_pred
    })
    
    # Создаем папку results, если ее нет, и сохраняем результаты в CSV
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_emo.csv'), index=False)
    
    return result_df


if __name__ == '__main__':
    # Определяем базовую директорию проекта
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
    # Путь к файлу данных
    data_path = os.path.join(base_dir, 'data', 'texts_with_features.csv')
    
    # Читаем данные из CSV файла
    df = pd.read_csv(data_path)
    
    # Обучаем модель и получаем предсказания
    predictions_df = train_and_predict(df)
    
    print(predictions_df)
