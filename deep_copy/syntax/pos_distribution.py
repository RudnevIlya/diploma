# pos_distribution.py
from razdel import tokenize
import pymorphy2
from collections import Counter

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def analyze_pos_distribution(text):
    """
    Анализирует распределение частей речи (POS) в тексте.
    
    Для каждого токена:
      - Выполняется лемматизация и определяется POS с помощью pymorphy2.
      - Группируются следующие категории:
          • NOUN: существительные
          • VERB: глаголы (теги VERB и INFN)
          • PRONOUN: местоимения (обычно имеют тег NPRO)
          • OTHER: все остальные найденные части речи
          
    Возвращает словарь с абсолютными значениями и процентами для каждой категории,
    а также общее число анализируемых токенов.
    """
    # Токенизация текста, отбираем только слова, состоящие из букв
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    if not tokens:
        return {}
    
    total_tokens = 0
    pos_counts = Counter()
    
    for token in tokens:
        token_lower = token.lower()
        parsed = morph.parse(token_lower)
        if not parsed:
            continue
        # Используем первое разборное значение
        pos = parsed[0].tag.POS
        if pos is None:
            continue
        total_tokens += 1
        
        # Группировка: для глаголов объединяем VERB и INFN
        if pos in ["VERB", "INFN"]:
            pos_counts["VERB"] += 1
        elif pos == "NOUN":
            pos_counts["NOUN"] += 1
        elif pos == "NPRO":  # местоимения
            pos_counts["PRONOUN"] += 1
        else:
            pos_counts["OTHER"] += 1

    distribution = {}
    # Вычисляем процент для каждой категории
    for category in ["NOUN", "VERB", "PRONOUN", "OTHER"]:
        count = pos_counts.get(category, 0)
        distribution[category] = {
            "count": count,
            "percentage": count / total_tokens if total_tokens > 0 else 0
        }
    
    distribution["TOTAL"] = total_tokens
    return distribution

if __name__ == '__main__':
    sample_text = (
        "Прекрасный день наступил, когда герой встал и пошёл в бой. "
        "Он понимал, что судьба зависит от его решений, и его сердце билось быстро, "
        "как молния. Он никогда не сомневался, что достигнет своей цели."
    )
    result = analyze_pos_distribution(sample_text)
    print("Распределение частей речи:")
    for pos, data in result.items():
        if pos != "TOTAL":
            print(f"{pos}: {data['count']} ({data['percentage']*100:.1f}%)")
    print(f"Всего токенов: {result.get('TOTAL', 0)}")
