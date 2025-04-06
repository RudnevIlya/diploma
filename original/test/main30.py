import os
import glob
import re
import numpy as np
import pymorphy2
import math
from razdel import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Инициализируем морфологический анализатор
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = [t.text for t in tokenize(text)]
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.strip()]
    return " ".join(lemmas)

def maas_ttr(text: str) -> float:
    tokens = text.split()
    N = len(tokens)
    V = len(set(tokens))
    if N == 0 or V == 0:
        return 0.0
    return (math.log(N) - math.log(V)) / (math.log(N) ** 2)

def normalize_maas(mttr: float, min_val=0.0, max_val=0.5) -> float:
    # Инвертируем и ограничиваем: чем меньше MTTR, тем выше разнообразие
    norm = (mttr - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, 1.0 - norm))

if __name__ == "__main__":
    folder = "./out"
    text_files = glob.glob(os.path.join(folder, "*.txt"))

    titles = []
    preprocessed_texts = []
    mttr_scores = []

    for file in text_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            title = os.path.splitext(os.path.basename(file))[0]
            preprocessed = preprocess_text(content)
            mttr = maas_ttr(preprocessed)
            norm_mttr = normalize_maas(mttr)

            titles.append(title)
            preprocessed_texts.append(preprocessed)
            mttr_scores.append(norm_mttr)

            print(f"{title} → Normalized MTTR: {norm_mttr:.4f}")

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
    avg_tfidf_scores = tfidf_matrix.mean(axis=1).A1

    print("\nРезультаты TF-IDF:")
    for title, score in zip(titles, avg_tfidf_scores):
        print(f"{title} → Средний TF-IDF: {score:.6f}")
