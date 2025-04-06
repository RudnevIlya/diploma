import os
import re
import math
import glob
import pandas as pd
import numpy as np
import pymorphy2
from razdel import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Инициализация морфоанализатора
morph = pymorphy2.MorphAnalyzer()

# Очистка и лемматизация
def preprocess_text(text: str) -> str:
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = [t.text for t in tokenize(text)]
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.strip()]
    return " ".join(lemmas)

# Maas TTR
def maas_ttr(text: str) -> float:
    tokens = text.split()
    N = len(tokens)
    V = len(set(tokens))
    if N == 0 or V == 0:
        return 0.0
    return (math.log(N) - math.log(V)) / (math.log(N) ** 2)

# Нормализация Maas TTR
def normalize_maas(mttr: float, min_val=0.0, max_val=0.5) -> float:
    norm = (mttr - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, 1.0 - norm))

# Загрузка CSV
df = pd.read_csv("results.csv")

# Подготовка
text_folder = "./out"
text_files = glob.glob(os.path.join(text_folder, "*.txt"))
title_to_text = {}

# Считываем тексты и сопоставляем с названиями
for file_path in text_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        title = os.path.splitext(os.path.basename(file_path))[0]
        title_to_text[title] = content

# Обработка: предобработка и метрики
preprocessed_texts = {}
mttr_scores = {}

for title, text in title_to_text.items():
    print(title)
    preprocessed = preprocess_text(text)
    mttr = maas_ttr(preprocessed)
    norm_mttr = normalize_maas(mttr)
    preprocessed_texts[title] = preprocessed
    mttr_scores[title] = norm_mttr

# Вычисление TF-IDF (только по доступным текстам)
corpus = [preprocessed_texts[title] for title in preprocessed_texts]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
avg_tfidf = tfidf_matrix.mean(axis=1).A1
tfidf_scores = dict(zip(preprocessed_texts.keys(), avg_tfidf))

# Добавление метрик в DataFrame по названию
df["Normalized_MTTR"] = df["Title"].map(mttr_scores)
df["Average_TFIDF"] = df["Title"].map(tfidf_scores)

# Сохранение по желанию
df.to_csv("results_with_metrics.csv", index=False)
