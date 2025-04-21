# emotional_arc.py
import math
import numpy as np
from deeppavlov import build_model, configs
from razdel import sentenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2 (при необходимости)
morph = pymorphy2.MorphAnalyzer()

# import deeppavlov.configs.classifiers as cl_configs
# print(dir(cl_configs))

# Инициализируем модель тональности DeepPavlov RuBERT
sentiment_model = build_model(configs.classifiers.rusentiment_bert, download=True)



def split_text_into_segments(text, num_segments=10):
    """
    Разбивает текст на num_segments равных по длине частей.
    Деление производится по символам – это приближённый способ.
    """
    text = text.strip()
    total_length = len(text)
    segment_length = total_length // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        # Для последнего сегмента возьмем остаток текста
        end = total_length if i == num_segments - 1 else (i + 1) * segment_length
        segments.append(text[start:end])
    return segments

def get_segment_sentiment(segment):
    """
    Получает общую тональность сегмента с использованием модели RuBERT.
    Возвращает строковую метку: "positive", "neutral" или "negative".
    """
    sentiment = sentiment_model([segment])
    return sentiment[0] if sentiment else "neutral"

def map_sentiment_to_numeric(sentiment_label):
    """
    Преобразует строковую метку тональности в числовое значение:
      "positive" -> +1, "neutral" -> 0, "negative" -> -1.
    """
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    return mapping.get(sentiment_label.lower(), 0)

def analyze_emotional_arc(text, num_segments=10):
    """
    Анализирует изменение эмоциональной тональности текста по сегментам.
    
    Действия:
      - Разбивает текст на num_segments равных частей.
      - Получает тональность каждого сегмента через модель RuBERT.
      - Преобразует тональности в числовые значения.
      - Вычисляет амплитуду (разницу между максимальным и минимальным значением).
      - Определяет число смен полярности (переходы от положительного к отрицательному и наоборот).
    
    Возвращает словарь с метриками:
      • segment_sentiments: список меток сегментов (строковые).
      • numeric_sentiments: список числовых значений тональности.
      • amplitude: разница между максимальным и минимальным значениями.
      • polarity_switches: количество смен полярности.
      • segments: список сегментов (для справки).
    """
    segments = split_text_into_segments(text, num_segments)
    segment_sentiments = []
    numeric_sentiments = []
    
    # Получаем тональность каждого сегмента
    for seg in segments:
        sentiment_label = get_segment_sentiment(seg)
        segment_sentiments.append(sentiment_label)
        numeric_value = map_sentiment_to_numeric(sentiment_label)
        numeric_sentiments.append(numeric_value)
    
    amplitude = max(numeric_sentiments) - min(numeric_sentiments)
    
    polarity_switches = 0
    # Считаем переходы между положительным и отрицательным (игнорируя нейтральные сегменты)
    for i in range(1, len(numeric_sentiments)):
        prev = numeric_sentiments[i - 1]
        curr = numeric_sentiments[i]
        if prev != 0 and curr != 0 and (prev * curr < 0):
            polarity_switches += 1
    
    return {
        "segment_sentiments": segment_sentiments,
        "numeric_sentiments": numeric_sentiments,
        "amplitude": amplitude,
        "polarity_switches": polarity_switches,
        "segments": segments
    }

if __name__ == '__main__':
    sample_text = (
        "В начале произведения герои полны надежд и радости. "
        "Однако по мере развития сюжета возникают первые трудности, вызывающие растущее напряжение. "
        "Ситуация накаляется, и герои сталкиваются с серьезными препятствиями. "
        "Тем не менее, в самый разгар борьбы наступает момент разрядки, когда положение начинает улучшаться. "
        "Постепенно напряжение спадает, уступая место тихой задумчивости и смирению. "
        "В кульминационный момент сюжет достигает пика эмоционального напряжения, после чего наступает неожиданное преодоление кризиса. "
        "Заключительная часть произведения насыщена чувством глубокого удовлетворения и облегчения, завершая эмоциональную дугу."
    )
    arc_metrics = analyze_emotional_arc(sample_text, num_segments=10)
    print("Сегменты тональности:", arc_metrics["segment_sentiments"])
    print("Числовые значения тональности:", arc_metrics["numeric_sentiments"])
    print("Амплитуда тональности:", arc_metrics["amplitude"])
    print("Количество смен полярности:", arc_metrics["polarity_switches"])
