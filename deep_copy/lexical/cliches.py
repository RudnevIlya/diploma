# cliches.py
import re
from razdel import tokenize

def load_cliche_phrases(file_path):
    """
    Загружает список фразеологизмов и клише из файла.
    Каждая строка файла должна содержать одну клишированную фразу в нижнем регистре.
    Возвращает список фраз.
    """
    cliche_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                phrase = line.strip().lower()
                if phrase:
                    cliche_list.append(phrase)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Используется пустой список клише.")
    return cliche_list

def count_cliche_occurrences(text, cliche_list):
    """
    Считает общее количество вхождений клишированных выражений в тексте.
    Для каждого выражения из списка производится поиск с использованием регулярного выражения,
    чтобы учесть только целые вхождения (с границами слова).
    Возвращает суммарное количество найденных клишированных выражений.
    """
    total_occurrences = 0
    text_lower = text.lower()
    for phrase in cliche_list:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        matches = re.findall(pattern, text_lower)
        total_occurrences += len(matches)
    return total_occurrences

def compute_cliche_ratio(text, cliche_list):
    """
    Вычисляет долю клишированных выражений относительно общего числа токенов в тексте.
    Доля рассчитывается как:
         (количество вхождений клише) / (общее число токенов),
    где токенизация производится с помощью Razdel с отбором только алфавитных токенов.
    Если текст не содержит слов, возвращается 0.
    """
    # Токенизация текста и отбор только алфавитных токенов
    tokens = [token.text for token in tokenize(text) if isinstance(token.text, str) and token.text.isalpha()]
    
    # Если текст не содержит токенов, возвращаем 0
    if not tokens:
        return 0
    
    # Считаем количество вхождений клише
    cliche_occurrences = count_cliche_occurrences(text, cliche_list)
    
    # Возвращаем долю клише
    return cliche_occurrences / len(tokens)


def analyze_cliches(text, cliche_file="data/cliche_phrases.txt"):
    """
    Основная функция анализа устойчивых выражений и клише.
    
    Аргументы:
        - text: текст произведения для анализа.
        - cliche_file: путь к файлу со списком клише.
    
    Функция загружает список клишированных фраз, вычисляет их долю в тексте и возвращает результат
    в виде словаря с ключом "ClicheRatio".
    """
    cliche_list = load_cliche_phrases(cliche_file)
    ratio = compute_cliche_ratio(text, cliche_list)
    return {"ClicheRatio": ratio}

if __name__ == '__main__':
    # Пример использования модуля
    sample_text = (
        "Всякая река знает своё русло. "
        "Как говорится, старую книгу не переписывают. "
        "Такие клише часто встречаются в стандартных текстах."
    )
    result = analyze_cliches(sample_text)
    print("Анализ клишированности текста:", result)
