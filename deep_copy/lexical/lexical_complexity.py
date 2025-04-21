# lexical_complexity.py
from razdel import tokenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2 (при необходимости для будущего расширения)
morph = pymorphy2.MorphAnalyzer()

def count_syllables(word):
    """
    Приблизительный подсчет слогов в слове посредством подсчета гласных.
    Для русского языка гласными считаются: а, е, ё, и, о, у, ы, э, ю, я.
    Если гласных нет, возвращает 1.
    """
    vowels = "аеёиоуыэюя"
    lower_word = word.lower()
    syllable_count = sum(1 for char in lower_word if char in vowels)
    return syllable_count if syllable_count > 0 else 1

def compute_lexical_complexity(text, long_word_threshold=7):
    """
    Вычисляет метрики сложности лексики текста:
      - avg_word_length: средняя длина слова (в буквах),
      - avg_syllable_count: среднее количество слогов в слове,
      - long_words_ratio: доля слов с числом букв >= long_word_threshold.
      
    Использует Razdel для токенизации.
    """
    # Токенизируем текст, отбираем только алфавитные токены (слова)
    tokens = [token.text for token in tokenize(text)]
    words = [token for token in tokens if token.isalpha()]
    
    if not words:
        return {"avg_word_length": 0, "avg_syllable_count": 0, "long_words_ratio": 0}
    
    total_length = 0
    total_syllables = 0
    long_words_count = 0
    
    for word in words:
        word_length = len(word)
        syllables = count_syllables(word)
        total_length += word_length
        total_syllables += syllables
        if word_length >= long_word_threshold:
            long_words_count += 1
    
    avg_word_length = total_length / len(words)
    avg_syllable_count = total_syllables / len(words)
    long_words_ratio = long_words_count / len(words)
    
    return {
        "avg_word_length": avg_word_length,
        "avg_syllable_count": avg_syllable_count,
        "long_words_ratio": long_words_ratio
    }

def analyze_lexical_complexity(text, long_word_threshold=7):
    """
    Основная функция анализа сложности лексики.
    
    Принимает текст и порог для длинного слова (по умолчанию 7 букв).
    Возвращает словарь с тремя метриками:
      - avg_word_length,
      - avg_syllable_count,
      - long_words_ratio.
    """
    return compute_lexical_complexity(text, long_word_threshold)

if __name__ == '__main__':
    sample_text = ("Пример сложного текста для анализа средней сложности лексики. "
                   "Проверяем, насколько наглядно определяется средняя длина слов, "
                   "количество слогов и доля длинных слов в тексте.")
    metrics = analyze_lexical_complexity(sample_text)
    print("Метрики лексической сложности:")
    print("Средняя длина слова (букв):", metrics["avg_word_length"])
    print("Среднее количество слогов в слове:", metrics["avg_syllable_count"])
    print("Доля длинных слов:", metrics["long_words_ratio"])
