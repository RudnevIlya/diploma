import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Функция для чтения текста по Book ID
def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла с именем <Book ID>.txt.
    Приводит Book ID к целому числу, чтобы избежать формата '123.0.txt'.
    """
    try:
        book_id_int = int(float(book_id))
        book_id_str = str(book_id_int)
    except Exception as e:
        book_id_str = str(book_id)
    file_path = os.path.join(txt_path, f"{book_id_str}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Функция для загрузки текстов и рейтингов из CSV с прогресс-баром
def load_texts_and_ratings(csv_file, txt_path="data"):
    """
    Считывает CSV с метаданными (обязательные столбцы: "Book ID" и "Rating")
    и для каждой записи загружает соответствующий текст.
    Возвращает списки: book_ids, texts и массив рейтингов для записей,
    где текст успешно прочитан.
    """
    df = pd.read_csv(csv_file)
    book_ids = []
    texts = []
    ratings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Загрузка текстов"):
        bid = row["Book ID"]
        rating = row["Rating"]
        text = read_text(bid, txt_path)
        if text is not None:
            # Приводим Book ID к целому числу для единообразия
            try:
                bid_int = int(float(bid))
                book_ids.append(str(bid_int))
            except:
                book_ids.append(str(bid))
            texts.append(text)
            ratings.append(rating)
    return book_ids, texts, np.array(ratings)

# Функция для вычисления метрик
def evaluate_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    kendall_corr, _ = kendalltau(y_true, y_pred)
    ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Spearman correlation ρ:", spearman_corr)
    print("Kendall Tau:", kendall_corr)
    print("NDCG:", ndcg)
    return mse, rmse, mae, spearman_corr, kendall_corr, ndcg

# Функция визуализации предсказаний
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title("Predicted vs Actual Ratings")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.show()

# Основной блок
def main():
    csv_file = "./data/_merged_filtered.csv"
    print("Считываем тексты, Book IDs и рейтинги...")
    book_ids, texts, ratings = load_texts_and_ratings(csv_file, txt_path="data")
    print(f"Найдено текстов: {len(texts)}")
    
    # Определяем стоп-слова для русского языка
    russian_stopwords = stopwords.words("russian")
    print("Вычисляем TF-IDF признаки...")
    # Используем TF-IDF с ограничением по количеству признаков
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=russian_stopwords)
    # Здесь также можно добавить дополнительную очистку текста, если требуется
    X = vectorizer.fit_transform(texts)
    print("Размерность TF-IDF матрицы:", X.shape)
    
    # Сохранение TF-IDF матрицы в CSV
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.insert(0, "Book ID", book_ids)
    tfidf_file = "data/tfidf_features.csv"
    tfidf_df.to_csv(tfidf_file, index=False)
    print(f"TF-IDF признаки сохранены в {tfidf_file}")
    
    # Разбиваем данные на обучающую и тестовую выборки
    print("Разбиваем данные на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=42)
    print("Размер обучающей выборки:", X_train.shape)
    print("Размер тестовой выборки:", X_test.shape)
    
    # Строим простейшую нейросеть
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # Выход для регрессии
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # Добавляем early stopping для контроля переобучения
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    # Обучение модели с прогресс-баром от TensorFlow
    print("Обучаем нейросеть...")
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=1)
    
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test).flatten()
    
    # Вычисляем метрики модели
    print("\nМетрики модели (TF-IDF + нейросеть):")
    mse, rmse, mae, spearman_corr, kendall_corr, ndcg = evaluate_model_metrics(y_test, y_pred)
    
    # Визуализируем предсказания
    plot_predictions(y_test, y_pred)
    
    # Сохранение модели и TF-IDF в файлы
    os.makedirs("models", exist_ok=True)
    model.save("models/tfidf_nn_model.h5")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Обученная нейросеть и TF-IDF vectorizer сохранены.")

if __name__ == '__main__':
    main()
