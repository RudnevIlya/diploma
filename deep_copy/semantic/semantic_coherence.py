# semantic_coherence.py
from razdel import sentenize
from sentence_transformers import SentenceTransformer
import numpy as np
import statistics
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация модели для получения эмбеддингов. 
# Здесь используется "paraphrase-multilingual-MiniLM-L12-v2", которая поддерживает русский.
# При наличии специализированных моделей (например, RuBERT или RuSimCSE) можно указать другое имя модели.
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def analyze_semantic_coherence(text):
    """
    Вычисляет семантическую связность (когерентность) текста.
    
    Процедура:
      1. Разбивка текста на предложения с помощью razdel.sentenize.
      2. Получение эмбеддингов для каждого предложения с помощью SentenceTransformer.
      3. Вычисление косинусной схожести между эмбеддингами для каждой пары соседних предложений.
      4. Расчёт средней косинусной схожести, которая служит мерой когерентности.
    
    Аргументы:
      - text: строка с анализируемым текстом.
    
    Возвращает:
      Словарь с тремя ключами:
        • "avg_cosine_similarity": средняя косинусная схожесть между соседними предложениями,
        • "cosine_similarities": список значений косинусной схожести для каждой пары соседних предложений,
        • "sentence_count": общее число выделенных предложений.
    """
    # Разбиваем текст на предложения
    sentences = [s.text.strip() for s in sentenize(text)]
    
    if len(sentences) < 2:
        return {"avg_cosine_similarity": 0, "cosine_similarities": [], "sentence_count": len(sentences)}
    
    # Вычисляем эмбеддинги для каждого предложения
    embeddings = model.encode(sentences)
    
    # Вычисляем косинусную схожесть между соседними предложениями
    cosine_similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        cosine_similarities.append(sim)
    
    # Рассчитываем среднее значение косинусной схожести
    avg_sim = statistics.mean(cosine_similarities)
    
    return {
        "avg_cosine_similarity": avg_sim,
        "cosine_similarities": cosine_similarities,
        "sentence_count": len(sentences)
    }

if __name__ == '__main__':
    sample_text = (
        "Это пример текста для анализа когерентности. "
        "Каждое предложение должно иметь логическую связь с предыдущим. "
        "Высокая схожесть между соседними предложениями свидетельствует о связном изложении мыслей."
    )
    result = analyze_semantic_coherence(sample_text)
    print("Средняя косинусная схожесть:", result["avg_cosine_similarity"])
    print("Список схожестей между предложениями:", result["cosine_similarities"])
    print("Количество предложений:", result["sentence_count"])
