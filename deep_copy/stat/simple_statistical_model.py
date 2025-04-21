import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from razdel import sentenize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла с именем <Book ID>.txt, расположенного в папке txt_path.
    Приводит Book ID к целому числу, чтобы избежать имен вида "123.0.txt".
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

def compute_text_statistics(text):
    """
    Вычисляет простые статистические признаки из текста:
      - total_words: общее число слов (с использованием регулярного выражения)
      - total_characters: общее число символов
      - avg_sentence_length: среднее число слов в предложении (используя razdel.sentenize)
      - unique_ratio: доля уникальных слов (уникальных слов / total_words)
      - avg_word_frequency: средняя частота слова (total_words / число уникальных слов)
    """
    # Вычисляем слова. Используем \b\w+\b для нахождения слов (учитываются цифры и буквы)
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = set(words)
    num_unique = len(unique_words)
    avg_word_freq = total_words / num_unique if num_unique > 0 else 0
    unique_ratio = num_unique / total_words if total_words > 0 else 0
    
    # Разбиваем текст на предложения с использованием razdel.sentenize
    sentences = [s.text.strip() for s in sentenize(text)]
    total_sentences = len(sentences) if sentences else 1
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    
    # Общее число символов
    total_characters = len(text)
    
    return {
        "total_words": total_words,
        "total_characters": total_characters,
        "avg_sentence_length": avg_sentence_length,
        "unique_ratio": unique_ratio,
        "avg_word_frequency": avg_word_freq
    }

def load_texts_and_ratings(csv_file, txt_path="data"):
    """
    Считывает CSV с метаданными (с обязательными столбцами "Book ID" и "Rating")
    и для каждой записи загружает текст из соответствующего файла.
    Возвращает списки: book_ids, texts, и массив рейтингов (там, где текст найден).
    При этом используется tqdm для отображения прогресса.
    """
    df = pd.read_csv(csv_file)
    book_ids = []
    texts = []
    ratings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка текстов"):
        bid = row["Book ID"]
        rating = row["Rating"]
        text = read_text(bid, txt_path)
        if text is not None:
            try:
                bid_int = int(float(bid))
                book_ids.append(str(bid_int))
            except:
                book_ids.append(str(bid))
            texts.append(text)
            ratings.append(rating)
    return book_ids, texts, np.array(ratings)

def evaluate_model_metrics(y_true, y_pred):
    """
    Вычисляет метрики:
      - MSE, RMSE, MAE
      - Spearman correlation ρ
      - Kendall Tau
      - NDCG (требует 2D массив)
    """
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

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title("Predicted vs Actual Ratings")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.show()

def main():
    csv_file = "./data/_merged_filtered.csv"
    print("Считываем тексты, Book IDs и рейтинги...")
    book_ids, texts, ratings = load_texts_and_ratings(csv_file, txt_path="data")
    print(f"Найдено текстов: {len(texts)}")
    
    # Для каждой записи вычисляем статистические признаки
    print("Вычисляем статистические признаки для каждого текста...")
    stats_list = []
    for text in tqdm(texts, desc="Вычисление статистик"):
        stats = compute_text_statistics(text)
        stats_list.append(stats)
    stats_df = pd.DataFrame(stats_list)
    stats_df["Rating"] = ratings
    stats_df["Book ID"] = book_ids
    output_csv = "data/simple_statistical_features.csv"
    stats_df.to_csv(output_csv, index=False)
    print(f"Вычисленные статистические признаки сохранены в {output_csv}")
    
    # Используем признаки: total_words, total_characters, avg_sentence_length, unique_ratio, avg_word_frequency
    feature_columns = ["total_words", "total_characters", "avg_sentence_length", "unique_ratio", "avg_word_frequency"]
    X = stats_df[feature_columns].values
    y = stats_df["Rating"].values
    
    # Разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Размер обучающей выборки:", X_train.shape)
    print("Размер тестовой выборки:", X_test.shape)
    
    # Обучаем модель линейной регрессии
    print("Обучаем модель линейной регрессии...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    print("Результаты модели на простых статистических признаках:")
    mse, rmse, mae, spearman_corr, kendall_corr, ndcg = evaluate_model_metrics(y_test, y_pred)
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
