# formality.py
from razdel import tokenize
import pymorphy2
import os

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def load_slang_list(file_path):
    """
    Загружает список жаргонных/разговорных слов из текстового файла.
    Каждая строка файла должна содержать одно слово (предполагается, что слова уже приведены к нижнему регистру).
    Возвращает множество слов.
    """
    slang_set = set()
    if not os.path.exists(file_path):
        # print(f"Файл {file_path} не найден. Жаргонный показатель не будет вычисляться.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                slang_set.add(word)
    return slang_set

def analyze_formality(text, slang_file=None):
    """
    Вычисляет показатели формальности стиля текста.
    
    Показатели:
      - pronoun_ratio: доля личных местоимений 1-го и 2-го лица (от общего числа слов)
      - adverb_adj_ratio: соотношение количества наречий к количеству прилагательных (ADVB / (ADJF+ADJS))
      - slang_ratio: доля слов из разговорного/жаргонного лексикона (если предоставлен slang_file)
    
    Возвращает словарь с вычисленными показателями.
    """
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {
            "pronoun_ratio": None,
            "adverb_adj_ratio": None,
            "slang_ratio": None
        }
    
    count_person = 0
    count_adverbs = 0
    count_adjectives = 0
    count_slang = 0
    
    slang_set = load_slang_list(slang_file) if slang_file else None

    for token in tokens:
        word_lower = token.lower()
        # Получаем морфологический разбор
        parsed = morph.parse(word_lower)
        if not parsed:
            continue
        # Берем первое (наиболее вероятное) разборное значение
        analysis = parsed[0]
        tag_str = str(analysis.tag).lower()
        
        # Определяем личные местоимения: tag.POS == "NPRO" и наличие индикатора 1-го или 2-го лица
        if analysis.tag.POS == "NPRO":
            # Приведем тег к строке для поиска индикаторов
            if "1л" in tag_str or "2л" in tag_str:
                count_person += 1
        
        # Наречия – обычно отмечаются как ADVB
        if analysis.tag.POS == "ADVB":
            count_adverbs += 1
        # Прилагательные: полная (ADJF) или краткая (ADJS)
        if analysis.tag.POS in ["ADJF", "ADJS"]:
            count_adjectives += 1
        
        # Если задан список жаргонных/разговорных слов, проверяем, есть ли слово в нем
        if slang_set is not None:
            if word_lower in slang_set:
                count_slang += 1

    pronoun_ratio = count_person / total_tokens
    # Если прилагательных нет, задаем ratio как None, иначе вычисляем
    adverb_adj_ratio = (count_adverbs / count_adjectives) if count_adjectives > 0 else None
    slang_ratio = (count_slang / total_tokens) if slang_set is not None else None

    return {
        "pronoun_ratio": pronoun_ratio,
        "adverb_adj_ratio": adverb_adj_ratio,
        "slang_ratio": slang_ratio
    }

if __name__ == '__main__':
    sample_text = (
        "Я думаю, что ты понимаешь, насколько это важно. "
        "Мы с тобой обсудим все детали, и, возможно, найдём лучшее решение. "
        "В этом разговоре я использую простые слова, чтобы донести сложную мысль."
    )
    # Если у вас есть файл со списком жаргонных слов (например, colloquial_lexicon.txt), укажите его путь;
    # иначе параметр можно опустить.
    result = analyze_formality(sample_text, slang_file="data/colloquial_lexicon.txt")
    print("Показатели формальности:")
    print(f"Доля личных местоимений (1-е и 2-е лица): {result['pronoun_ratio']:.3f}")
    if result['adverb_adj_ratio'] is not None:
        print(f"Соотношение наречий к прилагательным: {result['adverb_adj_ratio']:.3f}")
    else:
        print("Соотношение наречий к прилагательным: нет данных (нет прилагательных)")
    if result['slang_ratio'] is not None:
        print(f"Доля жаргонных слов: {result['slang_ratio']:.3f}")
    else:
        print("Доля жаргонных слов: не вычисляется (файл со slang-лексикой не найден)")
