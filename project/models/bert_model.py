import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель CatBoostRegressor для предсказания рейтинга художественного текста
    на основе 768 BERT-признаков и возвращает датафрейм с предсказаниями.
    
    Параметры:
    df : pd.DataFrame
        Датафрейм, содержащий столбцы:
          - BERT-признаки с именами "0", "1", ..., "767"
          - 'Rating' – целевая переменная
          - 'ID' – идентификатор объекта
        Остальные столбцы (например, Author, Title, AuthorRus, Value, Class, SimLow, SimMid, SimHigh)
        игнорируются.
        
    Возвращает:
    pd.DataFrame
        Датафрейм с колонками 'ID' и 'PredictedRating'.
        
    Дополнительно:
    - Выводит метрики MAE, RMSE и R².
    - Сохраняет предсказания в файл 'results/pred_bert.csv'.
    """
    # Выбираем только столбцы, представляющие BERT-признаки (названия – числа)
    features = sorted([col for col in df.columns if col.isdigit()], key=lambda x: int(x))
    if len(features) < 768:
        raise ValueError(f"Ожидалось 768 BERT признаков, получено {len(features)}")
    
    X = df[features]
    y = df['Rating']
    ids = df['ID']
    
    # Разбиваем данные на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )
    
    # Инициализируем CatBoostRegressor с поддержкой GPU (при наличии)
    try:
        model = CatBoostRegressor(task_type='GPU', random_seed=42, verbose=0)
        print("Используется GPU для CatBoostRegressor.")
    except Exception as e:
        print("GPU не доступно или возникла ошибка, используется CPU. Подробности:", e)
        model = CatBoostRegressor(random_seed=42, verbose=0)
    
    # Обучаем модель
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
    
    # Формируем итоговый DataFrame с ID и предсказанными значениями
    result_df = pd.DataFrame({
        'ID': id_test,
        'PredictedRating': y_pred
    })
    
    # Сохраняем результат в папку results под именем pred_bert.csv
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_bert.csv'), index=False)
    
    return result_df


import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает модель CatBoostRegressor для предсказания рейтинга художественного текста
    на основе 768 BERT-признаков и возвращает датафрейм с предсказаниями.

    Параметры:
    df_train : pd.DataFrame
        Датафрейм для обучения, содержащий столбцы:
          - BERT-признаки с именами "0", "1", ..., "767"
          - 'Rating' – целевая переменная
          - 'ID' – идентификатор объекта
        Остальные столбцы игнорируются.
    df_test : pd.DataFrame
        Датафрейм для тестирования с аналогичной структурой, на котором модель будет делать предсказания.

    Возвращает:
    pd.DataFrame
        Датафрейм с колонками 'ID' и 'PredictedRating'.

    Дополнительно:
    - Выводит метрики MAE, RMSE и R².
    - Сохраняет предсказания в файл 'results/pred_bert.csv'.
    """
    # Выбираем только столбцы, представляющие BERT-признаки (названия – числа)
    features = sorted([col for col in df_train.columns if col.isdigit()], key=lambda x: int(x))
    if len(features) < 768:
        raise ValueError(f"Ожидалось 768 BERT признаков, получено {len(features)}")
    
    # Формируем обучающие и тестовые выборки
    X_train = df_train[features]
    y_train = df_train['Rating']
    ids_train = df_train['ID']
    
    X_test = df_test[features]
    y_test = df_test['Rating']
    ids_test = df_test['ID']
    
    # Инициализируем CatBoostRegressor с поддержкой GPU (при наличии)
    try:
        model = CatBoostRegressor(task_type='GPU', random_seed=42, verbose=0)
        print("Используется GPU для CatBoostRegressor.")
    except Exception as e:
        print("GPU не доступно или возникла ошибка, используется CPU. Подробности:", e)
        model = CatBoostRegressor(random_seed=42, verbose=0)
    
    # Обучаем модель на обучающей выборке
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
    
    # Формируем итоговый DataFrame с ID и предсказанными значениями
    result_df = pd.DataFrame({
        'ID': ids_test,
        'PredictedRating': y_pred
    })
    
    # Сохраняем результат в папку results под именем pred_bert.csv
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(os.path.join('results', 'pred_bert.csv'), index=False)
    
    return result_df

if __name__ == '__main__':
    # Определяем базовую директорию проекта (предполагается, что этот скрипт лежит в папке models/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
    data_path = os.path.join(base_dir, 'data', 'texts_with_features.csv')


if __name__ == '__main__':
    # Определяем базовую директорию проекта (предполагается, что этот скрипт лежит в models/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
    data_path = os.path.join(base_dir, 'data', 'texts_with_features.csv')
    
    # Читаем данные из CSV
    df = pd.read_csv(data_path)
    
    # Обучаем модель и выводим предсказания
    predictions_df = train_and_predict(df)
    print(predictions_df)
