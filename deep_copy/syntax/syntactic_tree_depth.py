# syntactic_tree_depth.py
import stanza
import statistics

# Если модель для русского языка ещё не скачана, раскомментируйте следующую строку:
# stanza.download('ru')

# Инициализация пайплайна Stanza с необходимыми процессорами: tokenize, pos, depparse.
nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,depparse', use_gpu=True, verbose=False)


def compute_sentence_tree_depth(sentence):
    """
    Для данного предложения из Stanza вычисляет глубину (высоту) его dependency-дерева.
    
    Для каждого слова определяется число шагов, необходимое для достижения корневого узла (head == 0).
    Глубина предложения определяется как максимальное из этих чисел.
    """
    max_depth = 0
    words = sentence.words  # список объектов слова в предложении
    for word in words:
        depth = 1  # если слово является корнем, глубина = 1
        current_word = word
        # Продолжаем подниматься по дереву, пока не достигнем корня
        while current_word.head != 0:
            depth += 1
            # Предполагается, что список слов отсортирован по id, и родитель слова имеет индекс head-1
            parent_index = current_word.head - 1
            if parent_index < 0 or parent_index >= len(words):
                break
            current_word = words[parent_index]
        if depth > max_depth:
            max_depth = depth
    return max_depth

def analyze_tree_depth(text):
    """
    Анализирует текст: для каждого предложения получает глубину его синтаксического дерева,
    а затем вычисляет среднюю глубину по всем предложениям.
    
    Возвращает словарь с:
      - "avg_tree_depth": средняя глубина по предложениям,
      - "sentence_depths": список глубин для каждого предложения,
      - "sentence_count": количество предложений.
    """
    doc = nlp(text)
    sentence_depths = []
    for sentence in doc.sentences:
        depth = compute_sentence_tree_depth(sentence)
        sentence_depths.append(depth)
    if not sentence_depths:
        return {"avg_tree_depth": 0, "sentence_depths": [], "sentence_count": 0}
    avg_depth = statistics.mean(sentence_depths)
    return {
        "avg_tree_depth": avg_depth,
        "sentence_depths": sentence_depths,
        "sentence_count": len(sentence_depths)
    }

if __name__ == '__main__':
    sample_text = (
        "Я понял, что надо действовать быстро, потому что времени у нас нет. "
        "Когда наступила ночь, мы уже были далеко от дома, и дорога оказалась сложной."
    )
    metrics = analyze_tree_depth(sample_text)
    print("Средняя глубина синтаксического дерева:", metrics["avg_tree_depth"])
    print("Глубины для каждого предложения:", metrics["sentence_depths"])
    print("Количество предложений:", metrics["sentence_count"])
