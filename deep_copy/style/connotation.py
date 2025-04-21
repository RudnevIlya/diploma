# connotation.py
import os
import csv
from razdel import sentenize, tokenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def load_rusentilex(file_path):
    """
    Загружает RuSentiLex из файла rusentilex_2017.txt.
    
    Файл должен иметь формат с разделителями табуляции:
      token <tab> speech_part <tab> lemma <tab> sentiment <tab> source <tab> ambiguity
    Возвращает словарь, где ключ – лемма (в нижнем регистре), а значение – множество сентиментальных меток.
    """
    lexicon = {}
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return lexicon
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 6:
                continue
            # Берем token, speech_part, lemma, sentiment, source, ambiguity
            token, speech_part, lemma, sentiment, source, ambiguity = row[:6]
            lemma = lemma.strip().lower()
            sentiment = sentiment.strip().lower()
            if lemma:
                if lemma in lexicon:
                    lexicon[lemma].add(sentiment)
                else:
                    lexicon[lemma] = {sentiment}
    return lexicon

# Предопределённый список маркеров сарказма/иронии (в нижнем регистре)
SARCASM_MARKERS = [
    "ну конечно", "прям таки", "ага, конечно", "а ну да", "точно, как будто", 
    "смешно", "ах да", "вот именно", "ну, давай", "окей, конечно"
]

def check_sarcasm_markers(text):
    """
    Ищет в тексте маркеры сарказма и иронии из заранее заданного списка.
    Возвращает список найденных маркеров (без повторов).
    """
    found = set()
    lower_text = text.lower()
    for marker in SARCASM_MARKERS:
        if marker in lower_text:
            found.add(marker)
    return list(found)

def analyze_tonal_ambiguity(text, lexicon_file="data/rusentilex_2017.txt"):
    """
    Определяет для каждого предложения, содержит ли оно и положительные, и отрицательные сентиментальные метки.
    
    Текст разбивается на предложения (с помощью razdel.sentenize), затем каждое предложение токенизируется и лемматизируется.
    Если в одном предложении встречаются одновременно "positive" и "negative", оно считается тонально неоднозначным.
    
    Возвращает отношение числа неоднозначных предложений к общему числу предложений (тональную двусмысленность).
    """
    lexicon = load_rusentilex(lexicon_file)
    sentences = [s.text.strip() for s in sentenize(text)]
    if not sentences:
        return 0
    ambiguous_count = 0
    total = len(sentences)
    for sentence in sentences:
        tokens = [token.text for token in tokenize(sentence) if token.text.isalpha()]
        sentiments = set()
        for token in tokens:
            lemma = morph.parse(token.lower())[0].normal_form
            if lemma in lexicon:
                sentiments.update(lexicon[lemma])
        if "positive" in sentiments and "negative" in sentiments:
            ambiguous_count += 1
    return ambiguous_count / total if total > 0 else 0

def analyze_connotation(text, lexicon_file="data/rusentilex_2017.txt"):
    """
    Анализирует коннотации и подтекст текста:
      - sarcasm_markers: список найденных маркеров сарказма/иронии;
      - sarcasm_marker_count: количество таких маркеров;
      - tonal_ambiguity_score: доля предложений с обнаруженной тональной двусмысленностью.
    
    Возвращает словарь с результатами.
    """
    markers = check_sarcasm_markers(text)
    ambiguity_score = analyze_tonal_ambiguity(text, lexicon_file)
    return {
        "sarcasm_markers": markers,
        "sarcasm_marker_count": len(markers),
        "tonal_ambiguity_score": ambiguity_score
    }

if __name__ == '__main__':
    sample_text = (
        "Ну конечно, это же просто логично и предсказуемо. А, ну да, как всегда, всё идёт по плану! "
        "Всё это выглядит настолько абсурдно, что можно только усмехнуться."
    )
    result = analyze_connotation(sample_text, lexicon_file="data/rusentilex_2017.txt")
    print("Найденные маркеры сарказма/иронии:", result["sarcasm_markers"])
    print("Количество маркеров:", result["sarcasm_marker_count"])
    print("Тональная двусмысленность:", result["tonal_ambiguity_score"])
