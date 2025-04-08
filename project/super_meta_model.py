import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_meta_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает мета-регрессор для объединения предсказаний частных моделей
    и финального предсказания рейтинга.
    
    На входе:
      df : pd.DataFrame с колонками:
           - ID
           - Pred_Emotion
           - Pred_Lexical
           - Pred_Structure
           - Pred_BERT
           - Rating
           
    Возвращает:
      pd.DataFrame с колонками:
           - ID
           - PredictedRating
           
    Дополнительно:
      - Сохраняет итоговые предсказания в 'results/final_prediction.csv'
      - Выводит метрики MAE, RMSE, R²
      - Строит график Predicted vs Actual и сохраняет его в 'results/meta_model_plot.png'
    """
    # Выбираем признаки (предсказания частных моделей) и целевую переменную
    # features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_Structure', 'Pred_BERT']
    features = ["HD_D", "Average_TFIDF", "Emotional",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio"]
    X = df[features]
    y = df['Rating']
    ids = df['ID']
    
    # Разбиваем данные на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )
    
    # Обучаем мета-модель (LinearRegression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("Meta Model Metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    
    # Строим график Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Predicted vs Actual Rating")
    # Добавляем линию идеального предсказания
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    
    # Сохраняем график в папку results
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'meta_model_plot.png'))
    plt.show()
    
    # Формируем итоговый DataFrame с предсказаниями
    result_df = pd.DataFrame({
        'ID': id_test,
        'PredictedRating': y_pred
    })
    
    # Сохраняем итоговый DataFrame в CSV
    result_df.to_csv(os.path.join('results', 'final_prediction.csv'), index=False)
    
    return result_df


def train_meta_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает мета-регрессор для объединения предсказаний частных моделей
    и финального предсказания рейтинга.
    
    На входе:
      df : pd.DataFrame с колонками:
           - ID
           - Pred_Emotion
           - Pred_Lexical
           - Pred_Structure
           - Pred_BERT
           - Rating
           
    Возвращает:
      pd.DataFrame с колонками:
           - ID
           - PredictedRating
           
    Дополнительно:
      - Сохраняет итоговые предсказания в 'results/final_prediction.csv'
      - Выводит метрики MAE, RMSE, R²
      - Строит график Predicted vs Actual и сохраняет его в 'results/meta_model_plot.png'
    """
    # Выбираем признаки (предсказания частных моделей) и целевую переменную
    features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_Structure', 'Pred_BERT']
    features = ["HD_D", "Average_TFIDF", "Emotional",
    "AvgWordLength", "AvgSentenceLength", "LongWordRatio",
    "LexicalDiversity", "SentimentScore", "VerbRatio","AdjRatio","QuotesCount","NegationCount","ExclamationCount","AbstractNounRatio"]
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
    
    # Обучаем мета-модель (LinearRegression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("Meta Model Metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    
    # Строим график Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Predicted vs Actual Rating")
    # Добавляем линию идеального предсказания
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    
    # Сохраняем график в папку results
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'meta_model_plot.png'))
    plt.show()
    
    # Формируем итоговый DataFrame с предсказаниями
    result_df = pd.DataFrame({
        'ID': id_test,
        'PredictedRating': y_pred
    })
    
    # Сохраняем итоговый DataFrame в CSV
    result_df.to_csv(os.path.join('results', 'final_prediction.csv'), index=False)
    
    return result_df



if __name__ == '__main__':
    # Пример использования: предположим, что файл meta_input.csv содержит нужные столбцы
    # (ID, Pred_Emotion, Pred_Lexical, Pred_Structure, Pred_BERT, Rating)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
    data_path = os.path.join(base_dir, 'data', 'meta_model_input.csv')
    
    df = pd.read_csv(data_path)
    predictions_df = train_meta_model(df)
    print(predictions_df)
