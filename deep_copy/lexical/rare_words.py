# rare_words.py
from razdel import tokenize
import pymorphy2

def load_frequency_dictionary(file_path):
    """
    Загружает список самых частотных слов из файла.
    Каждый элемент файла должен содержать слово в нормальной форме.
    Возвращает множество слов.
    """
    freq_words = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    freq_words.add(word)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Будет использован пустой список частотных слов.")
    return freq_words

# Инициализация морфологического анализатора pymorphy2
morph = pymorphy2.MorphAnalyzer()

def compute_rare_words_percentage(text, freq_words):
    """
    Вычисляет процент слов, отсутствующих в списке частотных слов.
    
    Шаги:
      1. Токенизация текста с помощью Razdel.
      2. Отбор токенов, содержащих только буквы.
      3. Лемматизация слов с помощью pymorphy2.
      4. Подсчёт доли лемм, которых нет в предоставленном словаре.
    
    Если в тексте нет слов, возвращается 0.
    """
    tokens = [token.text for token in tokenize(text)]
    words = [token for token in tokens if token.isalpha()]
    
    if not words:
        return 0
    
    # Приведение каждого слова к его нормальной форме
    lemmas = [morph.parse(word.lower())[0].normal_form for word in words]
    
    total_words = len(lemmas)
    rare_count = sum(1 for lemma in lemmas if lemma not in freq_words)
    
    return rare_count / total_words

def analyze_rare_words(text, frequency_file="data/frequent_words.txt"):
    """
    Основная функция анализа редких и оригинальных слов.
    
    Аргументы:
      - text: строка с текстом произведения.
      - frequency_file: путь к файлу со списком частотных слов.
    
    Функция загружает частотный словарь, вычисляет процент слов, отсутствующих в нём,
    и возвращает результат в виде словаря с ключом "RareWordsPercentage".
    """
    freq_words = load_frequency_dictionary(frequency_file)
    rare_percentage = compute_rare_words_percentage(text, freq_words)
    return {"RareWordsPercentage": rare_percentage}

if __name__ == '__main__':
    # Пример использования модуля с тестовым текстом
    sample_text = (
        "Это пример текста для анализа оригинальности лексики. "
        "Некоторые слова могут отсутствовать в списке самых частотных, что указывает на оригинальность."
    )
    result = analyze_rare_words(sample_text)
    print("Анализ редких слов:", result)
