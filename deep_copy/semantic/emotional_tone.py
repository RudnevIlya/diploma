# emotional_tone.py
from razdel import tokenize
import pymorphy2
import csv
import os
from deeppavlov import build_model, configs

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def load_rusentilex(file_path):
    """
    Загружает RuSentiLex из файла rusentilex_2017.txt.
    
    Файл имеет формат с разделителем табуляции:
      token <tab> speech_part <tab> lemma <tab> sentiment <tab> source <tab> ambiguity
    Возвращает словарь, где ключ — лемма (строка), а значение — множество сентиментальных меток.
    """
    lexicon = {}
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден. Эмоциональный анализ не будет произведён.")
        return lexicon

    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # Убедимся, что строка содержит все необходимые поля (минимум 6)
            if len(row) < 6:
                continue
            token, speech_part, lemma, sentiment, source, ambiguity = row[:6]
            lemma = lemma.strip().lower()
            sentiment = sentiment.strip().lower()
            # При желании можно фильтровать нейтральные пометки, но здесь учитываем все
            if lemma:
                if lemma in lexicon:
                    lexicon[lemma].add(sentiment)
                else:
                    lexicon[lemma] = {sentiment}
    return lexicon

def analyze_emotional_profile(text, lexicon_file="data/rusentilex_2017.txt"):
    """
    Анализирует эмоциональный профиль текста с использованием RuSentiLex.
    
    Токенизирует текст с помощью Razdel и лемматизирует каждое слово. Если лемма встречается в лексиконе,
    увеличивает счётчик соответствующих сентиментальных меток.
    
    Возвращает словарь с распределением сентиментальных категорий и общее число слов.
    """
    lexicon = load_rusentilex(lexicon_file)
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    total_words = len(tokens)
    emotion_counts = {}
    
    for token in tokens:
        token_lower = token.lower()
        lemma = morph.parse(token_lower)[0].normal_form
        if lemma in lexicon:
            for sentiment in lexicon[lemma]:
                emotion_counts[sentiment] = emotion_counts.get(sentiment, 0) + 1
    
    return {"emotional_profile": emotion_counts, "total_words": total_words}

def analyze_overall_sentiment(text):
    """
    Анализирует общую тональность текста с использованием модели DeepPavlov RuBERT-тональности.
    
    Модель загружается через DeepPavlov и возвращает метку тональности:
    "positive", "negative" или "neutral".
    """
    # sentiment_model = build_model(configs.classifiers.rubert_sentiment, download=True)
    sentiment_model = build_model(configs.classifiers.rusentiment_bert, download=True)

    sentiment = sentiment_model([text])
    return sentiment[0] if sentiment else "neutral"

def analyze_emotional_tone(text, lexicon_file="data/rusentilex_2017.txt"):
    """
    Основная функция анализа эмоциональности и тона текста.
    
    Вычисляет:
      - overall_sentiment: общая тональность текста (positive, negative, neutral)
      - emotional_profile: распределение по сентиментальным меткам на основе RuSentiLex
      - total_words: общее число слов в тексте
    Возвращает словарь с результатами.
    """
    overall_sentiment = analyze_overall_sentiment(text)
    profile = analyze_emotional_profile(text, lexicon_file)
    result = {
        "overall_sentiment": overall_sentiment,
        "emotional_profile": profile["emotional_profile"],
        "total_words": profile["total_words"]
    }
    return result

if __name__ == '__main__':
    sample_text = (
        "Этот текст наполнен радостью и восторгом, но в нем присутствует и тень печали. "
        "Герой испытывал смесь эмоций: гнев, страх и удивление. "
        "Светлые моменты сменялись мрачными, вызывая глубокие переживания."
    )
    result = analyze_emotional_tone(sample_text, lexicon_file="data/rusentilex_2017.txt")
    print("Общая тональность текста:", result["overall_sentiment"])
    print("Эмоциональный профиль (по сентиментам):", result["emotional_profile"])
    print("Общее число слов в тексте:", result["total_words"])
