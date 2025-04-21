# punctuation_features.py
from razdel import sentenize

def analyze_punctuation(text):
    """
    Анализирует пунктуационные характеристики текста:
      - avg_commas_per_sentence: среднее число запятых на предложение;
      - exclamation_count: общее число восклицательных знаков;
      - question_count: общее число вопросительных знаков;
      - ellipsis_count: общее число эллипсисов («…»);
      - sentence_count: общее число предложений в тексте.

    Для расчёта средней запятых на предложение текст разбивается на предложения с помощью razdel.sentenize.
    """
    # Разбиваем текст на предложения
    sentences = list(sentenize(text))
    sentence_count = len(sentences)
    
    # Считаем запятые в каждом предложении
    total_commas = 0
    for sentence in sentences:
        total_commas += sentence.text.count(',')
    
    # Среднее число запятых на предложение
    avg_commas_per_sentence = total_commas / sentence_count if sentence_count > 0 else 0
    
    # Подсчёт восклицательных знаков
    exclamation_count = text.count('!')
    
    # Подсчёт вопросительных знаков
    question_count = text.count('?')
    
    # Подсчёт эллипсисов (символ «…»)
    ellipsis_count = text.count('…')
    
    return {
        "avg_commas_per_sentence": avg_commas_per_sentence,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "ellipsis_count": ellipsis_count,
        "sentence_count": sentence_count
    }

if __name__ == '__main__':
    sample_text = (
        "Какой сегодня чудесный день, не правда ли? "
        "Я не могу поверить, что такое случилось! "
        "Он задумался… потом ответил: «Я знаю, что делать, – сказал он, – и это будет удивительно». "
        "Но всё получилось иначе, как он ожидал."
    )
    metrics = analyze_punctuation(sample_text)
    print("Среднее количество запятых на предложение:", metrics["avg_commas_per_sentence"])
    print("Количество восклицательных знаков:", metrics["exclamation_count"])
    print("Количество вопросительных знаков:", metrics["question_count"])
    print("Количество эллипсисов:", metrics["ellipsis_count"])
    print("Количество предложений:", metrics["sentence_count"])
