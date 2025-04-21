# lexical_density.py
from razdel import tokenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def compute_lexical_density(text):
    """
    Вычисляет лексическую плотность текста как долю содержательных слов.
    
    Содержательные слова – это:
      - существительные (NOUN),
      - глаголы (VERB, INFN),
      - прилагательные (ADJF, ADJS),
      - наречия (ADVB).
      
    Функция токенизирует текст с использованием Razdel и для каждого слова с помощью pymorphy2
    определяет часть речи. Результат – отношение числа содержательных слов к общему количеству слов.
    Если текст не содержит слов, возвращается 0.
    """
    # Токенизируем текст
    tokens = [token.text for token in tokenize(text)]
    # Отбираем только алфавитные токены (слова)
    words = [token for token in tokens if token.isalpha()]
    
    if not words:
        return 0

    # Определяем набор тегов для содержательных слов
    content_tags = {"NOUN", "VERB", "INFN", "ADJF", "ADJS", "ADVB"}
    
    content_count = 0
    for word in words:
        # Приводим слово к нижнему регистру и получаем первое разборное значение
        parsed = morph.parse(word.lower())
        if parsed:
            pos = parsed[0].tag.POS  # Получаем часть речи
            if pos in content_tags:
                content_count += 1

    # Лексическая плотность – это доля содержательных слов среди всех
    lexical_density = content_count / len(words)
    return lexical_density

def analyze_text_lexical_density(text):
    """
    Основная функция анализа лексической плотности текста.
    
    Принимает строку с текстом и возвращает словарь с рассчитанной метрикой.
    """
    density = compute_lexical_density(text)
    return {"LexicalDensity": density}

if __name__ == '__main__':
    # Пример использования
    sample_text = "Пример текста для анализа. Здесь есть существительные, глаголы, прилагательные и наречия."
    result = analyze_text_lexical_density(sample_text)
    print("Результат анализа лексической плотности:", result)
