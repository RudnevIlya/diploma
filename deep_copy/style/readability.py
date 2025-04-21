# readability.py
from razdel import tokenize, sentenize
import pymorphy2

# Инициализируем морфологический анализатор для возможных расширений (например, если потребуется лемматизация)
morph = pymorphy2.MorphAnalyzer()

def count_syllables(word):
    """
    Приблизительный подсчет слогов в слове посредством подсчета гласных.
    Для русского языка гласными считаются: а, е, ё, и, о, у, ы, э, ю, я.
    Если гласных нет, возвращается 1.
    """
    vowels = "аеёиоуыэюя"
    word = word.lower()
    count = sum(1 for char in word if char in vowels)
    return count if count > 0 else 1

def get_words(text):
    """
    Возвращает список слов из текста.
    Здесь используются токены, состоящие из букв.
    """
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    return tokens

def get_sentences(text):
    """
    Возвращает список предложений из текста с помощью razdel.sentenize.
    """
    sentences = [s.text.strip() for s in sentenize(text)]
    return sentences

def compute_flesch_index(text):
    """
    Вычисляет адаптированный индекс читаемости Флеша для русского языка.
    
    Формула:
      FRE = 206.835 - 1.3 * (words/sentences) - 60 * (syllables/words)
    """
    sentences = get_sentences(text)
    if not sentences:
        return None
    words = get_words(text)
    if not words:
        return None
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables(word) for word in words)
    
    fre = 206.835 - 1.3 * (total_words / total_sentences) - 60 * (total_syllables / total_words)
    return fre

def compute_fog_index(text, complex_threshold=3):
    """
    Вычисляет индекс сложности Fog Index для текста.
    
    Формула:
      Fog = 0.4 * ((words/sentences) + 100*(complex_words/words))
    Слово считается сложным, если число слогов >= complex_threshold.
    """
    sentences = get_sentences(text)
    if not sentences:
        return None
    words = get_words(text)
    if not words:
        return None
    total_words = len(words)
    total_sentences = len(sentences)
    # Считаем сложные слова
    complex_words = [word for word in words if count_syllables(word) >= complex_threshold]
    num_complex = len(complex_words)
    
    fog = 0.4 * ((total_words / total_sentences) + 100 * (num_complex / total_words))
    return fog

if __name__ == '__main__':
    sample_text = (
        "Это пример сложного текста для анализа читаемости. "
        "Чем больше слов и слогов в предложении, тем ниже индекс читаемости, что может указывать на высокий уровень литературности. "
        "Однако слишком сложный текст может быть труден для восприятия."
    )
    flesch = compute_flesch_index(sample_text)
    fog = compute_fog_index(sample_text)
    print(f"Индекс читаемости Флеша: {flesch:.2f}")
    print(f"Индекс сложности Fog: {fog:.2f}")
