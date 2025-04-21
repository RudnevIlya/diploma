# syntactic_constructions.py
import re
from razdel import tokenize, sentenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def analyze_syntactic_constructions(text):
    """
    Анализирует текст на предмет употребления специфических синтаксических конструкций:
      - Причастных оборотов: формы с метками PRTS или PRTF;
      - Деепричастных оборотов: формы с меткой GRND;
      - Инфинитивных конструкций: формы с меткой INF;
      - Прямой речи: выявляется по наличию реплик в кавычках («...»)
        или строк, начинающихся с тире (— или –).

    Возвращает словарь со следующими полями:
      - participial_count: число найденных форм (PRTS или PRTF)
      - gerund_count: число найденных деепричастий (GRND)
      - infinitive_count: число инфинитивов (INF)
      - participial_gerund_ratio: отношение (participial_count + gerund_count) к числу токенов
      - infinitive_ratio: отношение инфинитивов к числу токенов
      - direct_speech_count: число обнаруженных реплик прямой речи
      - direct_speech_ratio: отношение числа реплик прямой речи к числу предложений
      - total_tokens: общее число токенизированных слов (исключая неалфавитные)
      - sentence_count: число предложений (по razdel.sentenize)
    """

    # Токенизация текста для морфологического анализа
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    total_tokens = len(tokens)

    participial_count = 0
    gerund_count = 0
    infinitive_count = 0

    for token in tokens:
        token_lower = token.lower()
        parsed = morph.parse(token_lower)
        if not parsed:
            continue
        tag = parsed[0].tag
        pos = tag.POS  # это строка, например, "NOUN", "PRTS", "GRND", "INF" и т.д.
        if pos in ["PRTS", "PRTF"]:
            participial_count += 1
        elif pos == "GRND":
            gerund_count += 1
        elif pos == "INFN":
            infinitive_count += 1


    # Анализ прямой речи.
    # Шаблон для кавычек (типичные русские кавычки: «...»)
    pattern_quotes = r'«[^»]+»'
    # Шаблон для реплик, начинающихся с тире (используя многострочный режим)
    pattern_dash = r'(?m)^(—|–)\s+'
    direct_speech_quotes = re.findall(pattern_quotes, text)
    direct_speech_dashes = re.findall(pattern_dash, text)
    direct_speech_count = len(direct_speech_quotes) + len(direct_speech_dashes)

    # Разбивка на предложения для нормализации прямой речи по числу предложений
    sentences = list(sentenize(text))
    sentence_count = len(sentences)

    # Вычисляем относительные показатели
    participial_gerund_ratio = ((participial_count + gerund_count) / total_tokens) if total_tokens else 0
    infinitive_ratio = (infinitive_count / total_tokens) if total_tokens else 0
    direct_speech_ratio = (direct_speech_count / sentence_count) if sentence_count else 0

    return {
        "participial_count": participial_count,
        "gerund_count": gerund_count,
        "infinitive_count": infinitive_count,
        "participial_gerund_ratio": participial_gerund_ratio,
        "infinitive_ratio": infinitive_ratio,
        "direct_speech_count": direct_speech_count,
        "direct_speech_ratio": direct_speech_ratio,
        "total_tokens": total_tokens,
        "sentence_count": sentence_count
    }

if __name__ == '__main__':
    sample_text = (
        "Глядя на закат, он решил, что нужно остановиться и отдохнуть. "
        "«Какой прекрасный вид!» – воскликнул он, – «никогда не видел ничего подобного». "
        "Движение остановилось, чтобы дать возможность насладиться моментом."
    )
    metrics = analyze_syntactic_constructions(sample_text)
    print("Анализ синтаксических конструкций:")
    print(f"Причастия: {metrics['participial_count']}")
    print(f"Деепричастия: {metrics['gerund_count']}")
    print(f"Инфинитивы: {metrics['infinitive_count']}")
    print(f"Отношение причастий+деепричастий к токенам: {metrics['participial_gerund_ratio']:.3f}")
    print(f"Отношение инфинитивов к токенам: {metrics['infinitive_ratio']:.3f}")
    print(f"Прямая речь (общее число): {metrics['direct_speech_count']}")
    print(f"Прямая речь (на предложение): {metrics['direct_speech_ratio']:.3f}")
    print(f"Всего токенов: {metrics['total_tokens']}")
    print(f"Количество предложений: {metrics['sentence_count']}")
