# syntactic_complexity.py
import stanza
import statistics

# Если модель не была загружена ранее, её можно загрузить так:
# stanza.download('ru')

# Инициализируем пайплайн Stanza для русского языка (используются процессоры tokenize и pos)
nlp = stanza.Pipeline(lang='ru', processors='tokenize', verbose=False)

# Список союзов подчинения и относительных местоимений, сигнализирующих о наличии придаточного предложения.
# Этот список можно расширять по необходимости.
SUBORDINATING_MARKERS = {
    "что", "который", "какая", "какие", "какой", "какую", "какие", "чтобы",
    "пока", "хотя", "если", "поскольку", "так как", "как", "где", "когда", "куда", "откуда", "словно", "будто", "потому", "нежели", "подобно тому", "после того"
}

def analyze_syntactic_complexity(text):
    """
    Анализирует синтаксическую сложность текста по следующим метрикам:
      - complex_sentence_ratio: доля сложноподчинённых предложений
      - avg_subordinate_clauses: среднее число придаточных (выраженных через маркеры) на предложение
      - sentence_count: общее число предложений
    
    В тексте с использованием синтаксического анализа Stanza предложения выделяются,
    затем для каждого предложения подсчитываются токены, соответствующие заданным маркерам подчинительной связи.
    """
    # Обрабатываем текст через Stanza
    doc = nlp(text)
    sentence_count = len(doc.sentences)
    
    if sentence_count == 0:
        return {
            "complex_sentence_ratio": 0,
            "avg_subordinate_clauses": 0,
            "sentence_count": 0
        }
    
    complex_sentences = 0
    subordinate_counts = []
    
    # Для каждого предложения подсчитываем число маркеров подчинения
    for sentence in doc.sentences:
        count_markers = 0
        for word in sentence.words:
            token = word.text.lower()
            # Если токен входит в список маркеров, учитываем его
            if token in SUBORDINATING_MARKERS:
                count_markers += 1
        subordinate_counts.append(count_markers)
        if count_markers > 0:
            complex_sentences += 1
    
    # Доля сложноподчинённых предложений
    complex_sentence_ratio = complex_sentences / sentence_count
    
    # Среднее число придаточных (маркерных) на предложение
    avg_subordinate_clauses = sum(subordinate_counts) / sentence_count
    
    return {
        "complex_sentence_ratio": complex_sentence_ratio,
        "avg_subordinate_clauses": avg_subordinate_clauses,
        "sentence_count": sentence_count
    }

if __name__ == '__main__':
    sample_text = (
        "Я понял, что надо действовать быстро, потому что времени у нас меньше, чем кажется. "
        "Хотя погода была пасмурной, мы решили идти дальше. "
        "Она спросила, где находится ближайшая станция, и я ответил, что она за углом."
    )
    metrics = analyze_syntactic_complexity(sample_text)
    print("Доля сложноподчинённых предложений:", metrics["complex_sentence_ratio"])
    print("Среднее число придаточных на предложение:", metrics["avg_subordinate_clauses"])
    print("Количество предложений:", metrics["sentence_count"])
