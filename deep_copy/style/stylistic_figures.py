# stylistic_figures.py
import re
import statistics
import numpy as np
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc
from gensim.models import KeyedVectors

# Загрузка предобученной модели FastText для русского языка.
# Убедитесь, что файл "cc.ru.300.bin" скачан с официального сайта fastText и сохранён в папке data.
FASTTEXT_MODEL_PATH = "data/cc.ru.300.bin"
try:
    # fasttext_model = KeyedVectors.load_word2vec_format(FASTTEXT_MODEL_PATH, binary=True)
    from gensim.models.fasttext import load_facebook_model
    fasttext_model = load_facebook_model(FASTTEXT_MODEL_PATH)
except Exception as e:
    print(f"Ошибка загрузки FastText модели: {e}")
    fasttext_model = None

CONSONANTS = set("бвгджзклмнпрстфхцчшщ")

def extract_adj_noun_pairs(text, similarity_threshold=0.3):
    """
    Извлекает последовательные пары (прилагательное, существительное) из текста с использованием Natasha.
    Для каждой такой пары, если оба слова присутствуют в FastText, вычисляется косинусная схожесть.
    Если схожесть ниже similarity_threshold, пара считается потенциально метафоричной.
    
    Возвращает список кортежей (прилагательное, существительное, cosine_similarity).
    """
    segmenter = Segmenter()
    embedding = NewsEmbedding()  
    morph_tagger = NewsMorphTagger(embedding)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    pairs = []
    tokens = doc.tokens  # Используем doc.tokens вместо doc.words
    for i in range(len(tokens) - 1):
        token1 = tokens[i]
        token2 = tokens[i+1]
        if not (hasattr(token1, "morph") and hasattr(token2, "morph")):
            continue
        pos1 = token1.morph.tag.POS
        pos2 = token2.morph.tag.POS
        if pos1 in ["ADJF", "ADJS"] and pos2 == "NOUN":
            adj = token1.text.lower()
            noun = token2.text.lower()
            if fasttext_model and (adj in fasttext_model.key_to_index) and (noun in fasttext_model.key_to_index):
                sim = fasttext_model.similarity(adj, noun)
                if sim < similarity_threshold:
                    pairs.append((adj, noun, sim))
            else:
                pairs.append((adj, noun, None))
    return pairs

def compute_epithet_density(text):
    """
    Рассчитывает долю прилагательных (ADJF, ADJS) среди всех токенов, извлечённых с использованием Natasha.
    """
    segmenter = Segmenter()
    embedding = NewsEmbedding()
    morph_tagger = NewsMorphTagger(embedding)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    tokens = doc.tokens
    total = len(tokens)
    if total == 0:
        return None
    count_adj = sum(1 for token in tokens if hasattr(token, "morph") and token.morph.tag.POS in ["ADJF", "ADJS"])
    return count_adj / total

def compute_alliteration(text):
    """
    Определяет коэффициент аллитерации в тексте.
    Разбивает текст на слова с помощью регулярного выражения, затем для каждой пары соседних слов,
    начинающихся с одной и той же согласной, вычисляет отношение таких пар к общему числу пар.
    """
    words = re.findall(r'\b[а-яёА-ЯЁ]+\b', text.lower())
    if len(words) < 2:
        return None
    total_pairs = len(words) - 1
    count_allit = 0
    for i in range(total_pairs):
        first = words[i][0]
        second = words[i+1][0]
        if first in CONSONANTS and second in CONSONANTS and first == second:
            count_allit += 1
    return count_allit / total_pairs

def analyze_stylistic_figures(text, similarity_threshold=0.3):
    """
    Анализирует стилистические фигуры в тексте:
      - metaphor_pairs: список пар (прилагательное, существительное, cosine_similarity),
                         где схожесть ниже порога (прокси для метафор).
      - epithet_density: доля прилагательных от общего числа токенов.
      - alliteration_ratio: коэффициент аллитерации.
    
    Возвращает словарь с метриками.
    """
    results = {
        "metaphor_pairs": extract_adj_noun_pairs(text, similarity_threshold),
        "epithet_density": compute_epithet_density(text),
        "alliteration_ratio": compute_alliteration(text)
    }
    return results

if __name__ == '__main__':
    sample_text = (
        "Холодный ветер разносил зловещие звуки ночи. "
        "Страшная тьма окутывала древние руины, а мрачная фигура, как ледяной луч, пронзала горизонт. "
        "Яркое солнце едва освещало унылые улицы города, где каждый камень хранил древние тайны."
    )
    figures = analyze_stylistic_figures(sample_text, similarity_threshold=0.3)
    
    print("Анализ метафорических пар:")
    for pair in figures["metaphor_pairs"]:
        print(f"Прилагательное: {pair[0]}, Существительное: {pair[1]}, Схожесть: {pair[2]}")
    
    epithet = figures["epithet_density"]
    if epithet is not None:
        print(f"\nЭпитетическая плотность: {epithet:.3f}")
    else:
        print("\nЭпитетическая плотность: нет данных")
    
    allit = figures["alliteration_ratio"]
    if allit is not None:
        print(f"Коэффициент аллитерации: {allit:.3f}")
    else:
        print("Коэффициент аллитерации: нет данных")
