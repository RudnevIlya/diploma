import tensorflow as tf  # Сначала TensorFlow
import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
import torch
import pymorphy2
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from razdel import tokenize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Проверка доступности GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем устройство: {device}")

# Загружаем модель для эмбеддингов (используется модель для русского языка)
model_name = 'sberbank-ai/sbert_large_nlu_ru'
embedder = SentenceTransformer(model_name, device=device)

# Инициализируем анализатор для лемматизации
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = [t.text for t in tokenize(text)]
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.strip()]
    return " ".join(lemmas)

# Чтение всех .txt файлов в текущей папке
text_files = glob.glob("./out/*.txt")
documents = []
titles = []
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        documents.append(content)
        title = os.path.splitext(os.path.basename(file))[0]
        titles.append(title)

# Загрузка файла result.csv для сопоставления названия произведения и автора
results = pd.read_csv("results.csv")
title_to_author = dict(zip(results['Title'], results['Author']))
authors = [title_to_author.get(title, "Unknown") for title in titles]

# Предобрабатываем все документы
preprocessed_docs = [preprocess_text(doc) for doc in documents]

# Вычисляем TTR (Type-Token Ratio) для каждого документа
ttr_scores = [LexicalRichness(doc).ttr for doc in preprocessed_docs]

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
avg_tfidf = tfidf_matrix.mean(axis=1).A1

# Контекстные эмбеддинги
embeddings = embedder.encode(preprocessed_docs, convert_to_numpy=True)

# Объединяем признаки
features = np.array([
    np.concatenate([embeddings[i], [ttr_scores[i]], [avg_tfidf[i]]])
    for i in range(len(documents))
])

# Тренировка классификатора
X_train, X_test, y_train, y_test = train_test_split(features, authors, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, y_train)

# Оценка
accuracy = clf.score(X_test, y_test)
print(f"Точность на тестовой выборке: {accuracy:.2f}")

# Сохранение
joblib.dump(clf, "author_classifier.joblib")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

def estimate_originality(text: str) -> float:
    preprocessed = preprocess_text(text)
    emb = embedder.encode([preprocessed], convert_to_numpy=True)[0]
    lr_text = LexicalRichness(preprocessed)
    ttr_score = lr_text.ttr
    tfidf_vec = tfidf_vectorizer.transform([preprocessed])
    avg_tfidf_score = tfidf_vec.mean()
    feat = np.concatenate([emb, [ttr_score], [avg_tfidf_score]]).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    max_prob = np.max(probs)
    originality = (1 - max_prob) + ttr_score + avg_tfidf_score
    return originality

def normalize_scores(scores, reference=None):
    if reference is None:
        reference = scores
    min_val = np.min(reference)
    max_val = np.max(reference)
    return [(s - min_val) / (max_val - min_val) if max_val > min_val else 0.0 for s in scores]

# Подсчёт оригинальности для всех документов
originality_scores = [estimate_originality(doc) for doc in documents]

# Пример использования
if __name__ == "__main__":
    test_folder = "./test"
    test_files = glob.glob(os.path.join(test_folder, "*.txt"))

    test_titles = []
    test_originalities = []
    test_authors = []

    for file in test_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            score = estimate_originality(content)
            test_originalities.append(score)
            title = os.path.splitext(os.path.basename(file))[0]
            test_titles.append(title)
            test_authors.append(title_to_author.get(title, "Unknown"))
            print(f"{title}: оригинальность = {score:.4f}")

    # Объединяем обучающие и тестовые
    all_titles = titles + test_titles
    all_scores = originality_scores + test_originalities
    all_authors = authors + test_authors

    # Нормализация
    norm_scores = normalize_scores(all_scores)

    # Назначим каждому автору свой цвет
    unique_authors = list(set(all_authors))
    cmap = cm.get_cmap('tab20', len(unique_authors))
    author_to_color = {author: mcolors.to_hex(cmap(i)) for i, author in enumerate(unique_authors)}
    bar_colors = [author_to_color[auth] for auth in all_authors]

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.bar(all_titles, norm_scores, color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Originality Score (0–1)')
    plt.title('Оригинальность текстов по авторам')
    plt.tight_layout()

    # Добавим легенду
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
