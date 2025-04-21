# раздел Прагматические признаки
# narrative_perspective.py
from razdel import tokenize
import pymorphy2

# Инициализируем морфологический анализатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

def analyze_narrative_perspective(text):
    """
    Анализирует нарративную перспективу текста, определяя долю личных местоимений
    1-го, 2-го и 3-го лица.

    Текст токенизируется с использованием Razdel. Для каждого токена выполняется морфологический анализ
    с помощью pymorphy2. Если слово является личным местоимением (NPRO) и содержит информацию о лице,
    оно учитывается в соответствующей категории:
      - 1-е лицо: «1per» (например, "я", "мы")
      - 2-е лицо: «2per» (например, "ты", "вы")
      - 3-е лицо: «3per» (например, "он", "она", "они")

    Возвращает словарь с:
      - first_person_count: количество местоимений 1-го лица
      - second_person_count: количество местоимений 2-го лица
      - third_person_count: количество местоимений 3-го лица
      - total_personal_pronouns: общее число личных местоимений
      - first_person_ratio: доля 1-го лица (first_person_count / total_personal_pronouns)
      - second_person_ratio: доля 2-го лица
      - third_person_ratio: доля 3-го лица
    """
    tokens = [token.text for token in tokenize(text) if token.text.isalpha()]
    first_count = 0
    second_count = 0
    third_count = 0

    for token in tokens:
        lower_token = token.lower()
        parsed = morph.parse(lower_token)
        if not parsed:
            continue
        analysis = parsed[0]
        if analysis.tag.POS == "NPRO":
            tag_str = str(analysis.tag).lower()  # пример: "npro,1per,sing,nom"
            if "1per" in tag_str:
                first_count += 1
            elif "2per" in tag_str:
                second_count += 1
            elif "3per" in tag_str:
                third_count += 1

    total = first_count + second_count + third_count
    result = {
        "first_person_count": first_count,
        "second_person_count": second_count,
        "third_person_count": third_count,
        "total_personal_pronouns": total
    }
    if total > 0:
        result["first_person_ratio"] = first_count / total
        result["second_person_ratio"] = second_count / total
        result["third_person_ratio"] = third_count / total
    else:
        result["first_person_ratio"] = 0
        result["second_person_ratio"] = 0
        result["third_person_ratio"] = 0

    return result

if __name__ == '__main__':
    sample_text = (
        "Я думаю, что это важно, и мы должны работать вместе. "
        "Ты понимаешь, о чем я говорю? "
        "Он же всегда уверен в своих силах, а они сомневаются."
    )
    perspective = analyze_narrative_perspective(sample_text)
    print("Нарративная перспектива:")
    for key, value in perspective.items():
        print(f"{key}: {value}")
