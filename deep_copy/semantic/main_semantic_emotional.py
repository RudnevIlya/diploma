# main_semantic_emotional.py
import os
import math
import pandas as pd
from tqdm import tqdm

# Импорт модулей для семантических и эмоциональных признаков.
from emotional_tone import analyze_emotional_tone
from emotional_arc import analyze_emotional_arc
from semantic_coherence import analyze_semantic_coherence
from semantic_originality import analyze_semantic_originality

def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла с именем <Book ID>.txt,
    расположенного в папке txt_path.
    Приводит Book ID к целому числу, чтобы избежать имен вида "123.0.txt".
    """
    try:
        # Приводим book_id к числовому значению, затем к целому и преобразуем обратно в строку.
        book_id_int = int(float(book_id))
        book_id_str = str(book_id_int)
    except Exception as e:
        book_id_str = str(book_id)
    
    file_path = os.path.join(txt_path, f"{book_id_str}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def flatten_emotional_profile(emotional_profile):
    """
    Преобразует эмоциональный профиль (словарь, где ключ – эмоция, значение – счёт)
    в словарь с префиксом "emotion_".
    """
    return {f"emotion_{emotion}": count for emotion, count in emotional_profile.items()}

def compute_features(text):
    """
    Вычисляет все семантические и эмоциональные признаки для одного текста.
    Возвращает объединённый словарь с метриками.
    """
    features = {}
    
    # Эмоциональная тональность
    emotional_tone = analyze_emotional_tone(text, lexicon_file="data/rusentilex_2017.txt")
    features["overall_sentiment"] = emotional_tone.get("overall_sentiment", None)
    features["total_words_emotional"] = emotional_tone.get("total_words", None)
    if "emotional_profile" in emotional_tone:
        features.update(flatten_emotional_profile(emotional_tone["emotional_profile"]))
    
    # Эмоциональная арка
    arc = analyze_emotional_arc(text, num_segments=10)
    features["arc_amplitude"] = arc.get("amplitude", None)
    features["arc_polarity_switches"] = arc.get("polarity_switches", None)
    
    # Семантическая связность
    coherence = analyze_semantic_coherence(text)
    features["avg_cosine_similarity"] = coherence.get("avg_cosine_similarity", None)
    features["sentence_count"] = coherence.get("sentence_count", None)
    
    # Семантическая оригинальность
    originality = analyze_semantic_originality(text)
    features["avg_adjacent_similarity"] = originality.get("avg_similarity", None)
    features["std_adjacent_similarity"] = originality.get("std_similarity", None)
    features["num_adjacent_pairs"] = originality.get("num_pairs", None)
    
    return features

def process_chunk(chunk, txt_path="data"):
    # Считываем текст для каждой записи
    chunk["text"] = chunk["Book ID"].apply(lambda x: read_text(x, txt_path))
    # Отбрасываем записи без текста и сбрасываем индекс
    chunk = chunk.dropna(subset=["text"]).reset_index(drop=True)
    
    features_list = []
    for idx, row in chunk.iterrows():
        text = row["text"]
        try:
            feats = compute_features(text)
        except Exception as e:
            print(f"Ошибка при обработке Book ID {row['Book ID']}: {e}")
            feats = {}
        features_list.append(feats)
    
    features_df = pd.DataFrame(features_list).reset_index(drop=True)
    # Убираем колонку с текстом, чтобы не сохранять её в итоговом файле
    chunk = chunk.drop(columns=["text"])
    final_chunk = pd.concat([chunk, features_df], axis=1)
    return final_chunk

def main():
    input_csv = "./data/_merged_filtered_v1.csv"
    output_csv = "data/semantic_emotional_metrics_v2.csv"
    chunk_size = 20  # Количество записей в одном чанке

    # Если итоговый CSV существует, удаляем его для нового запуска
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Подсчёт общего количества строк (без заголовка) для определения числа чанков
    with open(input_csv, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    total_chunks = math.ceil(total_lines / chunk_size)
    
    # Обрабатываем CSV по чанкам, обернув итератор в tqdm для визуализации прогресса.
    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)
    for chunk_idx, chunk in enumerate(tqdm(chunk_iter, total=total_chunks, desc="Обработка чанков"), start=1):
        final_chunk = process_chunk(chunk, txt_path="data")
        # Записываем результаты в CSV (append mode)
        if not os.path.exists(output_csv):
            final_chunk.to_csv(output_csv, index=False, mode='w')
        else:
            final_chunk.to_csv(output_csv, index=False, mode='a', header=False)
    
    print(f"Семантические и эмоциональные признаки сохранены в {output_csv}")

if __name__ == '__main__':
    main()
