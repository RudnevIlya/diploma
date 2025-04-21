# main_stylistic.py
import os
import math
import pandas as pd
from tqdm import tqdm
import re

# Импорт функций из модулей стилистических признаков
from readability import compute_flesch_index, compute_fog_index
from formality import analyze_formality
from stylistic_figures import analyze_stylistic_figures
from connotation import analyze_connotation
from narrative_perspective import analyze_narrative_perspective
from dialog_description import analyze_dialog_ratio

def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла с именем <Book ID>.txt, расположенного в папке txt_path.
    Приводит Book ID к целому числу, чтобы избежать имен вида "123.0.txt".
    """
    try:
        book_id_int = int(float(book_id))
        book_id_str = str(book_id_int)
    except Exception as e:
        book_id_str = str(book_id)
    file_path = os.path.join(txt_path, f"{book_id_str}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def compute_features(text):
    """
    Вычисляет стилистические признаки для одного текста.
    Возвращает объединённый словарь с метриками и их нормализованными значениями.
    """
    features = {}
    
    # Вычисляем общее число слов в тексте (будет базой для нормализации)
    total_words = len(re.findall(r'\b\w+\b', text, re.UNICODE))
    features["total_words"] = total_words  # базовый показатель (при необходимости)
    
    # Читаемость: индекс Флеша и Fog Index (уже имеют собственную шкалу)
    fre = compute_flesch_index(text)
    fog = compute_fog_index(text)
    features["readability_flesch"] = fre
    features["readability_fog"] = fog
    
    # Формальность стиля
    formal = analyze_formality(text, slang_file="data/colloquial_lexicon.txt")
    features.update(formal)
    
    # Стилистические фигуры
    figures = analyze_stylistic_figures(text, similarity_threshold=0.3)
    metaphor_count = len(figures.get("metaphor_pairs", []))
    features["metaphor_pair_count"] = metaphor_count
    # Нормализуем: количество метафорических пар на 1000 слов
    features["metaphor_pair_ratio"] = (metaphor_count / total_words * 1000) if total_words > 0 else 0
    
    features["epithet_density"] = figures.get("epithet_density", None)
    features["alliteration_ratio"] = figures.get("alliteration_ratio", None)
    
    # Коннотации и подтекст
    connot = analyze_connotation(text, lexicon_file="data/rusentilex_2017.txt")
    sarcasm_markers_count = connot.get("sarcasm_marker_count", 0)
    features["sarcasm_marker_count"] = sarcasm_markers_count
    # Нормализуем – количество маркеров на 1000 слов
    features["sarcasm_marker_ratio"] = (sarcasm_markers_count / total_words * 1000) if total_words > 0 else 0
    features["tonal_ambiguity_score"] = connot.get("tonal_ambiguity_score", 0)
    markers = connot.get("sarcasm_markers", [])
    features["sarcasm_markers"] = ";".join(markers) if markers else ""
    
    # Нарративная перспектива (уже в виде долей)
    perspective = analyze_narrative_perspective(text)
    features.update(perspective)
    
    # Соотношение диалогов и описания (отношение уже вычислено как dialog_ratio)
    dialog = analyze_dialog_ratio(text)
    features["dialog_total_words"] = dialog.get("total_words", None)
    features["dialog_words"] = dialog.get("dialogue_words", None)
    features["dialog_ratio"] = dialog.get("dialogue_ratio", None)
    
    return features
def process_chunk(chunk, txt_path="data"):
    """
    Обрабатывает один чанк DataFrame:
      - Считывает тексты по столбцу "Book ID"
      - Вычисляет стилистические признаки для каждой записи (с обработкой ошибок)
      - Возвращает DataFrame с исходными данными и вычисленными признаками.
    """
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
    chunk = chunk.drop(columns=["text"])
    final_chunk = pd.concat([chunk, features_df], axis=1)
    return final_chunk

def main():
    input_csv = "./data/_merged_filtered.csv"
    output_csv = "data/stylistic_metrics.csv"
    chunk_size = 10  # Количество записей в одном чанке
    
    # Если итоговый CSV существует, удаляем его для нового запуска
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Определяем общее число строк (без заголовка) для расчёта числа чанков
    with open(input_csv, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    total_chunks = math.ceil(total_lines / chunk_size)
    
    # Обрабатываем CSV по чанкам, с визуализацией через tqdm
    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)
    for chunk_idx, chunk in enumerate(tqdm(chunk_iter, total=total_chunks, desc="Обработка чанков"), start=1):
        final_chunk = process_chunk(chunk, txt_path="data")
        if not os.path.exists(output_csv):
            final_chunk.to_csv(output_csv, index=False, mode='w')
        else:
            final_chunk.to_csv(output_csv, index=False, mode='a', header=False)
    
    print(f"Стилистические признаки сохранены в {output_csv}")

if __name__ == '__main__':
    main()
