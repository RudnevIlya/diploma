# syntactic_features.py
from razdel import sentenize, tokenize
import statistics

def analyze_sentence_length(text):
    """
    Анализирует текст, вычисляя синтаксические метрики:
      - Средняя длина предложения в словах.
      - Стандартное отклонение длины предложений.
      
    Шаги:
      1. Разбивает текст на предложения с помощью razdel.sentenize.
      2. Для каждого предложения с помощью razdel.tokenize выбираются только алфавитные токены (слова).
      3. Вычисляется количество слов в каждом предложении.
      4. Рассчитываются среднее и стандартное отклонение.
      
    Если предложение одно или текста нет, стандартное отклонение приравнивается к 0.
    
    Возвращает словарь с метриками.
    """
    # Разбивка текста на предложения
    sentences = list(sentenize(text))
    if not sentences:
        return {"avg_sentence_length": 0, "std_sentence_length": 0, "sentence_count": 0}
    
    # Для каждого предложения вычисляем число слов (отбираем только алфавитные токены)
    sentence_lengths = []
    for sentence in sentences:
        words = [token.text for token in tokenize(sentence.text) if token.text.isalpha()]
        sentence_lengths.append(len(words))
    
    # Вычисляем среднюю длину и стандартное отклонение (если более одного предложения)
    avg_length = statistics.mean(sentence_lengths)
    std_length = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
    
    return {
        "avg_sentence_length": avg_length,
        "std_sentence_length": std_length,
        "sentence_count": len(sentences)
    }

if __name__ == '__main__':
    # Пример использования модуля
    sample_text = (
        "Это пример предложения для анализа. "
        "В классической прозе предложения могут быть длинными и сложными, содержащими множество элементов, "
        "в то время как упрощенный стиль характеризуется короткими фразами."
    )
    metrics = analyze_sentence_length(sample_text)
    print("Средняя длина предложения в словах:", metrics["avg_sentence_length"])
    print("Стандартное отклонение длины предложений:", metrics["std_sentence_length"])
    print("Количество предложений:", metrics["sentence_count"])
