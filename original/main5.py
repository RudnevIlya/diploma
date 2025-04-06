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

# Загрузка файла result.csv для сопоставления названия произведения и автора
results = pd.read_csv("results.csv")
title_to_author = dict(zip(results['Title'], results['Author']))

# Загрузка модели и векторизатора
if os.path.exists("author_classifier.joblib") and os.path.exists("tfidf_vectorizer.joblib"):
    clf = joblib.load("author_classifier.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    print("Загружены сохранённые модель и TF-IDF векторизатор.")
else:
    raise FileNotFoundError("Файлы модели и векторизатора не найдены. Сначала обучите модель.")

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

    # Получим глобальные мин и макс по всем оценкам (включая тестовые и обучающие)
    reference = test_originalities[:]



    global_min = 0
    global_max = max(test_originalities)

    # Если есть глобальный минимум и максимум, нормализуем относительно них
    norm_scores = [(s - global_min) / (global_max - global_min) if global_max > global_min else 0.0 for s in test_originalities]

    # БЫЛО РАНЬШЕ
    # norm_scores = normalize_scores(test_originalities, reference=test_originalities)

    # Назначим каждому автору свой цвет
    unique_authors = list(set(test_authors))
    cmap = cm.get_cmap('tab20', len(unique_authors))
    author_to_color = {author: mcolors.to_hex(cmap(i)) for i, author in enumerate(unique_authors)}
    bar_colors = [author_to_color[auth] for auth in test_authors]

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.bar(test_titles, norm_scores, color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Originality Score (0–1)')
    plt.title('Оригинальность тестовых текстов по авторам')
    plt.tight_layout()

    # Добавим легенду
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
