import tensorflow as tf  # Сначала TensorFlow
import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
import torch
import nltk
# nltk.download('punkt')  # Загрузка нужных токенизаторов
# from nltk.tokenize import word_tokenize
import pymorphy2
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from razdel import tokenize
import re
import pymorphy2

# # from nltk.tokenize import word_tokenize
# tokens = word_tokenize("Привет, как дела?", language="russian")

# Проверка доступности GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем устройство: {device}")

# Загружаем модель для эмбеддингов (используется модель для русского языка)
model_name = 'sberbank-ai/sbert_large_nlu_ru'
embedder = SentenceTransformer(model_name, device=device)

# Инициализируем анализатор для лемматизации
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    """
    Функция предобработки текста:
    - Приводит текст к нижнему регистру.
    - Удаляет все символы, кроме русских букв и пробелов.
    - Токенизирует текст.
    - Лемматизирует каждое слово.
    Возвращает очищенный текст в виде строки.
    """
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()

    # Токенизация
    tokens = [t.text for t in tokenize(text)]

    # Лемматизация
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.strip()]

    return lemmas

# Чтение всех .txt файлов в текущей папке
text_files = glob.glob("./out/*.txt")
documents = []
titles = []
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        documents.append(content)
        # Имя файла без расширения считается названием произведения
        title = os.path.splitext(os.path.basename(file))[0]
        titles.append(title)

# Загрузка файла result.csv для сопоставления названия произведения и автора
results = pd.read_csv("results.csv")
# Создаём словарь: название произведения -> автор
title_to_author = dict(zip(results['Title'], results['Author']))

# Определяем автора для каждого текста
authors = []
for title in titles:
    if title in title_to_author:
        authors.append(title_to_author[title])
    else:
        authors.append("Unknown")

# Предобрабатываем все документы
preprocessed_docs = [preprocess_text(doc) for doc in documents]

# Вычисляем TTR (Type-Token Ratio) для каждого документа
ttr_scores = []
for doc in preprocessed_docs:
    lr = LexicalRichness(doc)
    ttr_scores.append(lr.ttr)

# Обучаем TfidfVectorizer на предобработанных документах для оценки редкости слов
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
# Вычисляем среднее значение TF-IDF для каждого документа
avg_tfidf = tfidf_matrix.mean(axis=1).A1  # Преобразование в массив numpy

# Получаем контекстные эмбеддинги для каждого документа (используем предобработанный текст)
embeddings = embedder.encode(preprocessed_docs, convert_to_numpy=True)

# Объединяем признаки: эмбеддинг, TTR и среднее TF-IDF
features = []
for i in range(len(documents)):
    feat = np.concatenate([embeddings[i], [ttr_scores[i]], [avg_tfidf[i]]])
    features.append(feat)
features = np.array(features)

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, authors, test_size=0.2, random_state=42)

# Обучаем классификатор авторства (LogisticRegression)
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, y_train)

# Опциональная оценка точности на тестовой выборке
accuracy = clf.score(X_test, y_test)
print(f"Точность на тестовой выборке: {accuracy:.2f}")

# Сохраняем обученную модель и TF-IDF векторизатор для последующего использования
joblib.dump(clf, "author_classifier.joblib")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

def estimate_originality(text: str) -> float:
    """
    Функция оценки стилистической оригинальности текста.
    
    Шаги:
    1. Предобработка текста.
    2. Вычисление эмбеддинга, TTR и среднего TF-IDF.
    3. Формирование признакового вектора.
    4. Получение вероятностного распределения от классификатора авторства.
       Низкая уверенность (низкий максимум вероятностей) интерпретируется как высокий уровень оригинальности.
    5. Дополнительно учитывается высокая TTR и редкость слов.
    
    Возвращает числовое значение оригинальности.
    """
    preprocessed = preprocess_text(text)
    # Вычисляем эмбеддинг для текста
    emb = embedder.encode([preprocessed], convert_to_numpy=True)[0]
    # Вычисляем TTR
    lr_text = LexicalRichness(preprocessed)
    ttr_score = lr_text.ttr
    # Вычисляем среднее TF-IDF с помощью обученного векторизатора
    tfidf_vec = tfidf_vectorizer.transform([preprocessed])
    avg_tfidf_score = tfidf_vec.mean()
    # Объединяем признаки в один вектор
    feat = np.concatenate([emb, [ttr_score], [avg_tfidf_score]])
    feat = feat.reshape(1, -1)
    # Получаем вероятности принадлежности к каждому из известных авторов
    probs = clf.predict_proba(feat)[0]
    max_prob = np.max(probs)
    # Формула оценки оригинальности: чем ниже уверенность классификатора, тем выше оригинальность.
    # Дополнительно прибавляем вклад TTR и среднего TF-IDF.
    originality = (1 - max_prob) + ttr_score + avg_tfidf_score
    return originality

# Пример использования функции оценки оригинальности
if __name__ == "__main__":
    sample_text = "Здесь разместите пример текста для оценки его стилистической оригинальности."
    score = estimate_originality(sample_text)
    print("Оценка оригинальности текста:", score)
