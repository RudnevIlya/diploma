import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def custom_tokenizer(text):
    """
    Токенизирует текст с помощью регулярного выражения,
    затем лемматизирует каждое слово с использованием pymorphy2 и возвращает список лемм.
    """
    # Находим слова (учитывая русские и латинские буквы)
    tokens = re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas

def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла с именем <Book ID>.txt,
    расположенного в папке txt_path. Приводит Book ID к целому числу.
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

def load_texts_and_ratings(csv_file, txt_path="data"):
    """
    Считывает CSV с метаданными и для каждой записи загружает текст из соответствующего файла.
    Возвращает список Book IDs, список текстов и массив рейтингов для записей, где текст найден.
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
    Вычисляет различные метрики:
      - MSE, RMSE, MAE
      - Spearman correlation ρ
      - Kendall Tau
      - NDCG (рассматриваем тестовый набор как один документ)
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

def plot_feature_importances(model, feature_names):
    # Для линейной регрессии используем коэффициенты модели
    importances = model.coef_
    indices = np.argsort(np.abs(importances))[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Feature Importances (Коэффициенты модели)")
    plt.xlabel("Коэффициент")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
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
    
    # Вычисляем TF-IDF признаки с использованием нашего кастомного токенизатора (лемматизация)
    print("Вычисляем TF-IDF признаки с лемматизацией...")
    russian_stopwords = stopwords.words("russian")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=russian_stopwords, tokenizer=custom_tokenizer)
    X = vectorizer.fit_transform(texts)
    print("Размерность TF-IDF матрицы:", X.shape)
    
    # Сохраняем TF-IDF матрицу в CSV файл
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.insert(0, "Book ID", book_ids)
    tfidf_file = "data/tfidf_features.csv"
    tfidf_df.to_csv(tfidf_file, index=False)
    print(f"TF-IDF признаки сохранены в {tfidf_file}")
    
    # Разделяем данные на обучающую и тестовую выборки
    print("Разбиваем данные на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=42)
    print("Размер обучающей выборки:", X_train.shape)
    print("Размер тестовой выборки:", X_test.shape)
    
    print("Обучаем модель линейной регрессии...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    mse, rmse, mae, spearman_corr, kendall_corr, ndcg = evaluate_model_metrics(y_test, y_pred)
    plot_feature_importances(lr, vectorizer.get_feature_names_out())
    plot_predictions(y_test, y_pred)

if __name__ == '__main__':
    main()
