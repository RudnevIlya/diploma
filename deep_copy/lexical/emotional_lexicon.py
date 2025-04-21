# emotional_lexicon.py
from razdel import tokenize
import pymorphy2

def load_emotional_lexicon(file_path):
    """
    Загружает тональный словарь с эмоциональной лексикой.
    Файл должен содержать по одной нормализованной эмоциональной лемме на строке.
    Возвращает множество лемм.
    """
    lexicon = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    lexicon.add(word)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Эмоциональный анализ не будет произведён.")
    return lexicon

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def compute_emotional_word_frequency(text, lexicon):
    """
    Вычисляет частоту эмоционально окрашенных слов в тексте.
    
    Шаги:
      - Токенизация текста с помощью Razdel.
      - Отбор токенов, содержащих только буквы (слова).
      - Лемматизация каждого слова.
      - Подсчет количества лемм, присутствующих в тональном словаре.
    
    Возвращает отношение количества эмоциональных слов к общему числу слов.
    Если текст не содержит слов, возвращается 0.
    """
    tokens = [token.text for token in tokenize(text)]
    words = [token for token in tokens if token.isalpha()]
    if not words:
        return 0
    
    # Приведение слов к нормальной форме
    lemmas = [morph.parse(word.lower())[0].normal_form for word in words]
    
    count_emotional = sum(1 for lemma in lemmas if lemma in lexicon)
    
    # Относительная частота эмоциональных слов
    frequency = count_emotional / len(words)
    return frequency

def analyze_emotional_lexicon(text, lexicon_file="data/emotional_words.txt"):
    """
    Основная функция анализа эмоционально окрашенной лексики.
    
    Аргументы:
      - text: текст произведения для анализа.
      - lexicon_file: путь к файлу с тональным словарем (по умолчанию "data/emotional_words.txt").
    
    Функция загружает тональный словарь, вычисляет частоту эмоциональных слов 
    и возвращает результат в виде словаря с ключом "EmotionalWordFrequency".
    """
    lexicon = load_emotional_lexicon(lexicon_file)
    frequency = compute_emotional_word_frequency(text, lexicon)
    return {"EmotionalWordFrequency": frequency}

if __name__ == '__main__':
    # Пример использования модуля
    sample_text = (
        "Этот текст наполнен радостью и восторгом. "
        "Он может быть восхитительным, великолепным, но также полон грусти и печали."
    )
    result = analyze_emotional_lexicon(sample_text)
    print("Анализ эмоционально окрашенной лексики:", result)
