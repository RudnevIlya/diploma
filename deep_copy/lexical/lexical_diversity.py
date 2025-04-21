# lexical_diversity.py
import re
from razdel import tokenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    """
    Производит токенизацию текста с использованием Razdel и лемматизацию с помощью pymorphy2.
    Возвращает список лемм — нормализованных форм слов.
    """
    # Токенизация текста
    tokens = [token.text for token in tokenize(text)]
    # Лемматизация: приводим токены к нижнему регистру, оставляем только алфавитные токены
    lemmas = [morph.parse(token.lower())[0].normal_form for token in tokens if token.isalpha()]
    return lemmas

def compute_ttr(lemmas):
    """
    Вычисляет Type-Token Ratio (TTR) – отношение числа уникальных лемм к общему числу лемм.
    """
    if not lemmas:
        return 0
    return len(set(lemmas)) / len(lemmas)

def compute_yule_k(lemmas):
    """
    Вычисляет индекс Юла (Yule’s K), отражающий распределение частот лемм.
    Формула: K = 10^4 * (Σ(f*(f-1))) / N^2,
    где f – частота леммы, N – общее число лемм.
    """
    if not lemmas:
        return 0
    freq = {}
    for word in lemmas:
        freq[word] = freq.get(word, 0) + 1
    N = len(lemmas)
    M2 = sum(f * (f - 1) for f in freq.values())
    yule_k = (10**4 * M2) / (N**2)
    return yule_k

def mtld_calc(lemmas, threshold=0.72):
    """
    Вычисляет MTLD (Measure of Textual Lexical Diversity).
    Алгоритм идёт по списку лемм (прямо и в обратном направлении), накапливая токены до тех пор,
    пока TTR не опустится ниже порога. Итоговое значение MTLD – отношение общего числа токенов к среднему числу сегментов.
    """
    def _mtld_calc(lemmas, threshold):
        factors = 0
        token_count = 0
        type_count = {}
        for word in lemmas:
            token_count += 1
            type_count[word] = type_count.get(word, 0) + 1
            current_ttr = len(type_count) / token_count
            if current_ttr <= threshold:
                factors += 1
                token_count = 0
                type_count = {}
        # Корректировка для неполного сегмента (partial factor)
        if token_count > 0:
            current_ttr = len(type_count) / token_count
            partial = (1 - current_ttr) / (1 - threshold)
        else:
            partial = 0
        return factors + partial

    factors_forward = _mtld_calc(lemmas, threshold)
    factors_reverse = _mtld_calc(list(reversed(lemmas)), threshold)
    if factors_forward + factors_reverse == 0:
        return -1
    mtld_value = len(lemmas) / ((factors_forward + factors_reverse) / 2)
    return mtld_value

def analyze_lexical_diversity(text):
    """
    Основная функция анализа лексического разнообразия:
      - токенизация и лемматизация текста,
      - вычисление TTR, MTLD и Yule’s K.
    Возвращает словарь с рассчитанными метриками.
    """
    lemmas = preprocess_text(text)
    ttr = compute_ttr(lemmas)
    mtld = mtld_calc(lemmas)
    yule_k = compute_yule_k(lemmas)
    return {"TTR": ttr, "MTLD": mtld, "YuleK": yule_k}
