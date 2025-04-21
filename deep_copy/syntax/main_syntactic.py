# main_syntactic_chunked.py
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# Импорт функций из модулей синтаксических метрик
from syntactic_features import analyze_sentence_length
from syntactic_complexity import analyze_syntactic_complexity
from syntactic_tree_depth import analyze_tree_depth
from punctuation_features import analyze_punctuation
from syntactic_constructions import analyze_syntactic_constructions

def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла, имя которого формируется как "<Book ID>.txt".
    """
    file_path = os.path.join(txt_path, f"{book_id}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def compute_all_syntactic_metrics(text):
    """
    Вычисляет все синтаксические метрики для переданного текста:
      - Длина и сложность предложений (analyze_sentence_length)
      - Сложноподчинённость (analyze_syntactic_complexity)
      - Глубина синтаксического дерева (analyze_tree_depth)
      - Пунктуационные особенности (analyze_punctuation)
      - Синтаксические конструкции (analyze_syntactic_constructions)
    Результаты объединяются в один словарь.
    """
    metrics = {}
    metrics.update(analyze_sentence_length(text))
    metrics.update(analyze_syntactic_complexity(text))
    metrics.update(analyze_tree_depth(text))
    metrics.update(analyze_punctuation(text))
    metrics.update(analyze_syntactic_constructions(text))
    return metrics

def process_chunk(chunk):
    """
    Обрабатывает один чанк DataFrame.
    Считывает тексты, вычисляет синтаксические метрики и добавляет результаты в виде отдельных столбцов.
    """
    # Считываем текст для каждого произведения
    chunk["text"] = chunk["Book ID"].apply(lambda bid: read_text(bid))
    # Отбрасываем записи, для которых текст не найден
    chunk = chunk.dropna(subset=["text"])

    # Вычисляем метрики и сразу преобразуем словари в отдельные столбцы
    metrics_series = chunk["text"].progress_apply(compute_all_syntactic_metrics)
    metrics_df = metrics_series.apply(pd.Series)
    
    # Объединяем исходный чанк с вычисленными метриками
    chunk = pd.concat([chunk.drop(columns=["text"]), metrics_df], axis=1)
    return chunk

def main():
    input_file = "./data/_merged_filtered.csv"
    output_file = "data/syntactic_text_metrics.csv"
    chunksize = 50  # можно настроить под объем данных
    
    # Удаляем старый выходной файл, если он существует
    if os.path.exists(output_file):
        os.remove(output_file)
    
    chunk_iter = pd.read_csv(input_file, chunksize=chunksize)
    first_chunk = True
    i = -1
    for chunk in tqdm(chunk_iter, desc="Обработка чанков"):
        i+=1
        if i<75: continue
        processed_chunk = process_chunk(chunk)
        # processed_chunk = processed_chunk.drop(columns=["text"])
        # Добавляем полученные данные к выходному файлу
        processed_chunk.to_csv(str.replace(output_file,".csv",f"{i}.csv"), mode='a', index=False, header=first_chunk)
        # if first_chunk:
        #     first_chunk = False

    print(f"Синтаксические метрики сохранены в файл {output_file}")

if __name__ == '__main__':
    main()
