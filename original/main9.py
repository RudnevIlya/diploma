import tensorflow as tf  # Сначала TensorFlow
import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
import torch
import pymorphy2
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from razdel import tokenize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math

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

def root_ttr(text: str) -> float:
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / (len(tokens) ** 0.5)

def estimate_originality(text: str):
    preprocessed = preprocess_text(text)
    emb = embedder.encode([preprocessed], convert_to_numpy=True)[0]
    ttr_score = root_ttr(preprocessed)
    tfidf_vec = tfidf_vectorizer.transform([preprocessed])
    avg_tfidf_score = tfidf_vec.mean()
    feat = np.concatenate([emb, [ttr_score], [avg_tfidf_score]]).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    max_prob = np.max(probs)
    originality = 3 * (1 - max_prob) + 0.2 * ttr_score + 8000 * avg_tfidf_score

    predicted_author = clf.classes_[np.argmax(probs)]
    return originality, predicted_author, (1 - max_prob), ttr_score, avg_tfidf_score

def normalize_scores(scores, reference=None):
    if reference is None:
        reference = scores
    min_val = np.min(reference)
    max_val = np.max(reference)
    return [(s - min_val) / (max_val - min_val) if max_val > min_val else 0.0 for s in scores]

# Загрузка файла result.csv для сопоставления названия произведения и автора
results = pd.read_csv("results.csv")
title_to_author = dict(zip(results['Title'], results['Author']))

# Проверка существования модели и векторизатора
model_exists = os.path.exists("author_classifier.joblib") and os.path.exists("tfidf_vectorizer.joblib")

if model_exists:
    clf = joblib.load("author_classifier.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    print("Загружены сохранённые модель и TF-IDF векторизатор.")
else:
    print("Файлы модели не найдены. Выполняется обучение модели...")

    text_files = glob.glob("./out/*.txt")
    documents = []
    titles = []
    for file in text_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(content)
            title = os.path.splitext(os.path.basename(file))[0]
            titles.append(title)

    authors = [title_to_author.get(title, "Unknown") for title in titles]
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    ttr_scores = [root_ttr(doc) for doc in preprocessed_docs]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
    avg_tfidf = tfidf_matrix.mean(axis=1).A1
    embeddings = embedder.encode(preprocessed_docs, convert_to_numpy=True)
    features = np.array([
        np.concatenate([embeddings[i], [ttr_scores[i]], [avg_tfidf[i]]])
        for i in range(len(documents))
    ])
    X_train, X_test, y_train, y_test = train_test_split(features, authors, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Точность на тестовой выборке: {accuracy:.2f}")
    joblib.dump(clf, "author_classifier.joblib")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

if __name__ == "__main__":
    test_folder = "./test"
    test_files = glob.glob(os.path.join(test_folder, "*.txt"))

    test_titles = []
    test_originalities = []
    test_authors = []
    predicted_authors = []
    all_uncertainty = []
    all_rttr = []
    all_tfidf = []

    for file in test_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            originality, predicted_author, p_uncertainty, p_ttr, p_tfidf = estimate_originality(content)
            test_originalities.append(originality)
            title = os.path.splitext(os.path.basename(file))[0]
            test_titles.append(title[:20])
            actual_author = title_to_author.get(title, "Unknown")
            test_authors.append(actual_author)
            predicted_authors.append(predicted_author)
            all_uncertainty.append(p_uncertainty)
            all_rttr.append(p_ttr)
            all_tfidf.append(p_tfidf)
            print(f"{title} → оригинальность: {originality:.4f}, неуверенность: {p_uncertainty:.4f}, RTTR: {p_ttr:.4f}, TF-IDF: {p_tfidf:.4f}, автор: {actual_author}, предсказано: {predicted_author}")

    combined_data = list(zip(test_authors, test_titles, test_originalities, predicted_authors, all_uncertainty, all_rttr, all_tfidf))
    combined_data.sort(key=lambda x: x[0])
    test_authors, test_titles, test_originalities, predicted_authors, all_uncertainty, all_rttr, all_tfidf = zip(*combined_data)

    unique_authors = list(set(test_authors))
    cmap = cm.get_cmap('tab20', len(unique_authors))
    author_to_color = {author: mcolors.to_hex(cmap(i)) for i, author in enumerate(unique_authors)}
    bar_colors = [author_to_color[auth] for auth in test_authors]

    # График 1 — только на основе неуверенности
    plt.figure(figsize=(10, 4))
    bars = plt.bar(test_titles, all_uncertainty, color=bar_colors)
    plt.title("Оригинальность (только неуверенность модели)")
    plt.ylabel("1 - max_prob")
    plt.xticks(rotation=45, ha='right')
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # График 2 — только RTTR + усиленный TF-IDF
    rttr_plus_tfidf = [r + tf * 1000 for r, tf in zip(all_rttr, all_tfidf)]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(test_titles, rttr_plus_tfidf, color=bar_colors)
    plt.title("Оригинальность (RTTR + усиленный TF-IDF)")
    plt.ylabel("RTTR + TF-IDF*1000")
    plt.xticks(rotation=45, ha='right')
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    global_min = min(test_originalities)
    global_max = max(test_originalities)
    norm_scores = [(s - global_min) / (global_max - global_min) if global_max > global_min else 0.0 for s in test_originalities]

    bar_colors = [author_to_color[auth] for auth in test_authors]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(test_titles, norm_scores, color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Originality Score (0–1)')
    plt.title('Оригинальность тестовых текстов по авторам')
    plt.tight_layout()

    for i, (author, pred) in enumerate(zip(test_authors, predicted_authors)):
        label = f"{author}\n({pred})"
        plt.text(i, norm_scores[i] + 0.01, label, ha='center', va='bottom', fontsize=8, rotation=90)

    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
