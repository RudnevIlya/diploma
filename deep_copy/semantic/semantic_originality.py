# semantic_originality.py
import os
import statistics
import numpy as np
from razdel import tokenize
import pymorphy2
from gensim.models import KeyedVectors

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

# Загрузка предобученной модели FastText для русского языка.
# Например, используйте модель "cc.ru.300.bin", которая должна быть сохранена в папке data.
fasttext_model_path = "data/cc.ru.300.bin"

try:
    fasttext_model = KeyedVectors.load_word2vec_format(fasttext_model_path, binary=True)
except Exception as e:
    print(f"Ошибка загрузки модели FastText из {fasttext_model_path}: {e}")
    fasttext_model = None

def preprocess_text(text):
    """
    Токенизирует текст с использованием Razdel и лемматизирует его с помощью pymorphy2.
    Возвращает список лемм (слова в начальной форме), отбирая только алфавитные токены.
    """
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    lemmas = [morph.parse(token.lower())[0].normal_form for token in tokens]
    return lemmas

def compute_adjacent_cosine_similarities(lemmas):
    """
    Вычисляет косинусную схожесть для каждой пары соседних слов в списке lemmas
    с использованием модели FastText.
    Если одно из слов отсутствует в словаре модели, пара пропускается.
    Возвращает список значений схожести.
    """
    similarities = []
    if fasttext_model is None:
        return similarities

    # Для каждой пары соседних лемм вычисляем схожесть
    for i in range(len(lemmas) - 1):
        word1, word2 = lemmas[i], lemmas[i+1]
        if word1 in fasttext_model.vocab and word2 in fasttext_model.vocab:
            sim = fasttext_model.similarity(word1, word2)
            similarities.append(sim)
    return similarities

def analyze_semantic_originality(text):
    """
    Анализирует семантическую оригинальность текста.
    
    Идея:
      - Текст токенизируется и лемматизируется.
      - Вычисляются косинусные схожести для соседних слов.
      - Низкая средняя схожесть может указывать на необычные, авторские комбинации слов,
        что может служить прокси для выявления метафор и оригинальных образов.
    
    Возвращает словарь с:
      • avg_similarity: средняя косинусная схожесть для пар соседних слов,
      • std_similarity: стандартное отклонение схожести,
      • num_pairs: количество обработанных пар.
    """
    lemmas = preprocess_text(text)
    similarities = compute_adjacent_cosine_similarities(lemmas)
    if not similarities:
        return {"avg_similarity": None, "std_similarity": None, "num_pairs": 0}
    avg_sim = statistics.mean(similarities)
    std_sim = statistics.stdev(similarities) if len(similarities) > 1 else 0
    return {"avg_similarity": avg_sim, "std_similarity": std_sim, "num_pairs": len(similarities)}

if __name__ == '__main__':
    sample_text = (
        "Солнце плакало золотыми слезами, а ветер шептал древние тайны. "
        "Необычные образы вспыхивали в воображении, словно искры на темном небе."
    )
    result = analyze_semantic_originality(sample_text)
    print("Средняя косинусная схожесть для смежных слов:", result["avg_similarity"])
    print("Стандартное отклонение схожести:", result["std_similarity"])
    print("Количество обработанных пар слов:", result["num_pairs"])
