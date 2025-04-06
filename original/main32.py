import os
import re
import math
import glob
import pandas as pd
import numpy as np
from collections import Counter
from razdel import tokenize
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import hypergeom

# Инициализация морфоанализатора
morph = pymorphy2.MorphAnalyzer()

# Очистка и лемматизация
def preprocess_text(text: str) -> str:
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = [t.text for t in tokenize(text)]
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.strip()]
    return " ".join(lemmas)

# HD-D: Hypergeometric Distribution Diversity
def hdd(text: str, sample_size: int = 42) -> float:
    tokens = text.split()
    N = len(tokens)
    if N < sample_size:
        return 0.0
    freqs = Counter(tokens)
    total = 0.0
    for freq in freqs.values():
        rv = hypergeom(N, freq, sample_size)
        prob = 1 - rv.pmf(0)
        total += prob
    return total / sample_size

# Загрузка CSV
df = pd.read_csv("results.csv")

# Путь к текстам
text_folder = "./out"
text_files = glob.glob(os.path.join(text_folder, "*.txt"))

# Словари для хранения текстов и метрик
title_to_text = {}
preprocessed_texts = {}
hdd_scores = {}

# Чтение и обработка файлов
for file_path in text_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        title = os.path.splitext(os.path.basename(file_path))[0]
        title_to_text[title] = content

        # Предобработка
        preprocessed = preprocess_text(content)
        preprocessed_texts[title] = preprocessed

        # HD-D
        print(title)
        hdd_scores[title] = hdd(preprocessed)

# TF-IDF по предобработанным текстам
corpus = list(preprocessed_texts.values())
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
avg_tfidf = tfidf_matrix.mean(axis=1).A1
tfidf_scores = dict(zip(preprocessed_texts.keys(), avg_tfidf))

# Добавление метрик в DataFrame
df["HD_D"] = df["Title"].map(hdd_scores)
df["Average_TFIDF"] = df["Title"].map(tfidf_scores)

# Сохранение при необходимости
df.to_csv("results_with_hdd.csv", index=False)
