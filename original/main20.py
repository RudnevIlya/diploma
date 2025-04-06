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
from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

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
    return originality, predicted_author, (1 - max_prob), ttr_score, avg_tfidf_score, len(preprocessed.split())

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

    print("Фильтрация текстов и балансировка по авторам...")

    print("Фильтрация текстов и балансировка по объёму (словам)...")

    text_files = glob.glob("./out/*.txt")
    author_to_texts = defaultdict(list)

    # Читаем тексты, фильтруем по длине и соответствию автору
    for file in text_files:
        title = os.path.splitext(os.path.basename(file))[0]
        if title not in title_to_author:
            continue
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                word_count = len(content.split())
                if word_count < 50:  # Игнорируем слишком короткие тексты
                    continue
                author = title_to_author[title]
                author_to_texts[author].append((title, content, word_count))
        except Exception as e:
            print(f"Ошибка при чтении {file}: {e}")

    print("\nОбъём доступных слов по авторам до балансировки:")
    for author, texts in author_to_texts.items():
        total_words = sum(t[2] for t in texts)
        num_texts = len(texts)
        print(f"  {author}: {total_words} слов в {num_texts} текстах")


    # Ограничим количество авторов, у которых есть достаточный объём текста
    min_total_words_required = 100000  # слов на автора

    eligible_authors = {a: ts for a, ts in author_to_texts.items()
                        if sum(t[2] for t in ts) >= min_total_words_required}

    if not eligible_authors:
        raise ValueError("Недостаточно данных: нет ни одного автора с нужным объёмом текста.")

    print(f"Выбрано авторов: {len(eligible_authors)} (по {min_total_words_required} слов)")

    balanced_documents = []
    balanced_titles = []
    balanced_authors = []

    for author, texts in eligible_authors.items():
        # Сортируем тексты по убыванию длины
        sorted_texts = sorted(texts, key=lambda x: -x[2])
        current_word_count = 0
        for title, content, word_count in sorted_texts:
            if current_word_count + word_count <= min_total_words_required:
                balanced_documents.append(content)
                balanced_titles.append(title)
                balanced_authors.append(author)
                current_word_count += word_count
            if current_word_count >= min_total_words_required:
                break
        print(f"{author}: использовано {current_word_count} слов")

    # Предобработка
    preprocessed_docs = [preprocess_text(doc) for doc in balanced_documents]
    ttr_scores = [root_ttr(doc) for doc in preprocessed_docs]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
    avg_tfidf = tfidf_matrix.mean(axis=1).A1
    embeddings = embedder.encode(preprocessed_docs, convert_to_numpy=True)

    # Финальная сборка признаков
    features = np.array([
        np.concatenate([embeddings[i], [ttr_scores[i]], [avg_tfidf[i]]])
        for i in range(len(balanced_documents))
    ])


    # from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # # Используем только эмбеддинги (без tf-idf и rttr)
    # embeddings_only = np.array([feat[:-2] for feat in features])
    # n_samples = len(embeddings_only)

    # # Автоматический выбор метода снижения размерности
    # if n_samples < 10:
    #     print("Мало данных, используем PCA для понижения размерности...")
    #     reducer = PCA(n_components=2)
    #     method_name = "PCA"
    # else:
    #     perplexity = min(30, max(5, n_samples // 3))
    #     print(f"Понижаем размерность с помощью t-SNE (perplexity={perplexity})...")
    #     reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    #     method_name = f"t-SNE (perplexity={perplexity})"

    # # Вычисление понижения размерности
    # reduced = reducer.fit_transform(embeddings_only)

    # # Визуализация
    # plt.figure(figsize=(10, 8))
    # unique_authors = sorted(set(balanced_authors))
    # palette = sns.color_palette("hls", len(unique_authors))
    # author_to_color = {a: palette[i] for i, a in enumerate(unique_authors)}
    # colors = [author_to_color[a] for a in balanced_authors]

    # plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=10, alpha=0.8)

    # # Подписи авторов в центрах их кластеров
    # for author in unique_authors:
    #     idx = [i for i, a in enumerate(balanced_authors) if a == author]
    #     x_mean = np.mean(reduced[idx, 0])
    #     y_mean = np.mean(reduced[idx, 1])
    #     plt.text(x_mean, y_mean, author, fontsize=10, weight='bold')

    # plt.title(f"Кластеризация авторов на основе эмбеддингов ({method_name})")
    # plt.xlabel("Компонента 1")
    # plt.ylabel("Компонента 2")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Извлекаем только эмбеддинги (без rttr и tf-idf)
    embeddings_only = np.array([feat[:-2] for feat in features])
    n_samples = len(embeddings_only)

    # Снижение размерности
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    pca = PCA(n_components=2)

    tsne_result = tsne.fit_transform(embeddings_only)
    pca_result = pca.fit_transform(embeddings_only)

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    unique_authors = sorted(set(balanced_authors))
    palette = sns.color_palette("hls", len(unique_authors))
    author_to_color = {a: palette[i] for i, a in enumerate(unique_authors)}
    colors = [author_to_color[a] for a in balanced_authors]

    # t-SNE
    axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=10, alpha=0.8)
    axes[0].set_title(f"t-SNE (perplexity={perplexity})")
    axes[0].set_xlabel("Компонента 1")
    axes[0].set_ylabel("Компонента 2")
    axes[0].grid(True)

    # PCA
    axes[1].scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=10, alpha=0.8)
    axes[1].set_title("PCA")
    axes[1].set_xlabel("Компонента 1")
    axes[1].set_ylabel("Компонента 2")
    axes[1].grid(True)

    # Подписи авторов на центрах кластеров
    for i, author in enumerate(unique_authors):
        idx = [j for j, a in enumerate(balanced_authors) if a == author]
        tsne_x = tsne_result[idx, 0].mean()
        tsne_y = tsne_result[idx, 1].mean()
        pca_x = pca_result[idx, 0].mean()
        pca_y = pca_result[idx, 1].mean()
        axes[0].text(tsne_x, tsne_y, author, fontsize=10, weight='bold')
        axes[1].text(pca_x, pca_y, author, fontsize=10, weight='bold')

    plt.suptitle("Сравнение t-SNE и PCA на эмбеддингах авторов", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



    # text_files = glob.glob("./out/*.txt")
    # documents = []
    # titles = []
    # authors = []

    # # Собираем тексты с учётом сопоставления с авторами
    # author_to_texts = defaultdict(list)
    # for file in text_files:
    #     title = os.path.splitext(os.path.basename(file))[0]
    #     if title not in title_to_author:
    #         continue
    #     try:
    #         with open(file, 'r', encoding='utf-8') as f:
    #             content = f.read().strip()
    #             if len(content.split()) < 50:  # Фильтруем слишком короткие тексты
    #                 continue
    #             author = title_to_author[title]
    #             author_to_texts[author].append((title, content))
    #     except Exception as e:
    #         print(f"Ошибка при чтении {file}: {e}")

    # for author, texts in author_to_texts.items():
    #     print(f"Автор: {author}, количество подходящих текстов: {len(texts)}")

    # # Обрезаем тексты, чтобы у всех авторов было одинаковое количество (балансировка)
    # # min_count = min(len(texts) for texts in author_to_texts.values())

    # min_required_texts = 3
    # author_to_texts = {author: texts for author, texts in author_to_texts.items() if len(texts) >= min_required_texts}

    # if not author_to_texts:
    #     raise ValueError("Недостаточно авторов с нужным количеством текстов. Нужно больше данных.")

    # min_count = min(len(texts) for texts in author_to_texts.values())
    # print(f"Используем по {min_count} текстов на автора")

    # print(f"Используем по {min_count} текстов на автора")

    # for author, texts in author_to_texts.items():
    #     for title, content in texts[:min_count]:
    #         titles.append(title)
    #         authors.append(author)
    #         documents.append(content)

    # # Предобработка
    # preprocessed_docs = [preprocess_text(doc) for doc in documents]
    # ttr_scores = [root_ttr(doc) for doc in preprocessed_docs]
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
    # avg_tfidf = tfidf_matrix.mean(axis=1).A1
    # embeddings = embedder.encode(preprocessed_docs, convert_to_numpy=True)

    # Финальная сборка признаков
    # features = np.array([
    #     np.concatenate([embeddings[i], [ttr_scores[i]], [avg_tfidf[i]]])
    #     for i in range(len(documents))
    # ])

    X_train, X_test, y_train, y_test = train_test_split(features, balanced_authors, test_size=0.2, random_state=42)
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
    text_lengths = []

    for file in test_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            originality, predicted_author, p_uncertainty, p_ttr, p_tfidf, length = estimate_originality(content)
            test_originalities.append(originality)
            title = os.path.splitext(os.path.basename(file))[0]
            test_titles.append(title[:20])
            actual_author = title_to_author.get(title, "Unknown")
            test_authors.append(actual_author)
            predicted_authors.append(predicted_author)
            all_uncertainty.append(p_uncertainty)
            all_rttr.append(p_ttr)
            all_tfidf.append(p_tfidf)
            text_lengths.append(length)
            print(f"{title} → оригинальность: {originality:.4f}, неуверенность: {p_uncertainty:.4f}, RTTR: {p_ttr:.4f}, TF-IDF: {p_tfidf:.4f}, автор: {actual_author}, предсказано: {predicted_author}")

    combined_data = list(zip(test_authors, test_titles, test_originalities, predicted_authors, all_uncertainty, all_rttr, all_tfidf, text_lengths))
    combined_data.sort(key=lambda x: x[0])
    test_authors, test_titles, test_originalities, predicted_authors, all_uncertainty, all_rttr, all_tfidf, text_lengths = zip(*combined_data)

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
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
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
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # График 3 — оригинальность в зависимости от длины текста (лог шкала по X)
    colors = [author_to_color[auth] for auth in test_authors]
    plt.figure(figsize=(10, 6))
    for author in unique_authors:
        xs = [length for a, length in zip(test_authors, text_lengths) if a == author]
        ys = [orig for a, orig in zip(test_authors, test_originalities) if a == author]
        plt.scatter(xs, ys, label=author, color=author_to_color[author])
    plt.xscale("log")
    plt.xlabel("Длина текста (логарифмическая шкала)")
    plt.ylabel("Оригинальность")
    plt.title("Зависимость оригинальности от длины текста")
    plt.legend(title="Авторы", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # График 4 — нормализованная оригинальность
    global_min = min(test_originalities)
    global_max = max(test_originalities)
    norm_scores = [(s - global_min) / (global_max - global_min) if global_max > global_min else 0.0 for s in test_originalities]

    bar_colors = [author_to_color[auth] for auth in test_authors]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(test_titles, norm_scores, color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Originality Score (0–1)')
    plt.title('Оригинальность тестовых текстов по авторам')
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in author_to_color.values()]
    labels = list(author_to_color.keys())
    plt.legend(handles, labels, title="Авторы", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
