# раздел Структурные признаки
# dialog_description.py
import re

def count_words(text):
    """
    Считает словообразующие символы (слова) в тексте.
    Используем регулярное выражение, учитывающее русские и латинские буквы.
    """
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', text, re.UNICODE)
    return len(words)

def extract_dialog_segments(text):
    """
    Извлекает фрагменты диалогов из текста двумя способами:
      1. Текст, заключённый в кавычки («…», “…” или "…").
      2. Строки, начинающиеся с тире или дефиса (предполагается, что это прямая речь).
    
    Возвращает список строк, содержащих диалоговые фрагменты.
    """
    # Извлекаем текст в кавычках (подход для различных типов кавычек)
    # Используем непоглощающий режим для минимального захвата.
    quotes = re.findall(r'[«“"](.+?)[»”"]', text, re.DOTALL)
    
    # Извлекаем строки, начинающиеся с тире или дефиса (учитываем возможные пробелы и перевод строки)
    dash_lines = re.findall(r'(?m)^\s*[-–]\s*(.+)', text)
    
    # Объединяем результаты
    dialogues = quotes + dash_lines
    return dialogues

def analyze_dialog_ratio(text):
    """
    Анализирует соотношение диалогов и описания.
    
    Вычисления:
      - total_words: общее число слов в тексте.
      - dialogue_words: число слов, содержащихся в извлечённых диалоговых фрагментах.
      - dialogue_ratio: отношение dialogue_words / total_words.
    
    Возвращает словарь с метриками.
    """
    total_words = count_words(text)
    dialogues = extract_dialog_segments(text)
    dialogue_text = " ".join(dialogues)
    dialogue_words = count_words(dialogue_text)
    
    ratio = dialogue_words / total_words if total_words > 0 else 0
    return {
        "total_words": total_words,
        "dialogue_words": dialogue_words,
        "dialogue_ratio": ratio
    }

if __name__ == '__main__':
    sample_text = (
        "«Привет, как дела?» — спросил он. "
        "Она ответила: «Все хорошо, спасибо». \n"
        "— Ну да, конечно, — добавил он, и продолжил свой рассказ. "
        "В то время как окружающий мир оставался тихим и неприметным."
    )
    
    result = analyze_dialog_ratio(sample_text)
    print("Общее число слов:", result["total_words"])
    print("Число слов в диалогах:", result["dialogue_words"])
    print("Доля слов в диалогах:", result["dialogue_ratio"])
