# main.py
import os
import pandas as pd
import datetime

# Импорт функций из ранее созданных модулей
from lexical_diversity import analyze_lexical_diversity
from lexical_density import analyze_text_lexical_density
from lexical_complexity import analyze_lexical_complexity
from rare_words import analyze_rare_words
from cliches import analyze_cliches
from emotional_lexicon import analyze_emotional_lexicon

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import stanza
stanza.download('ru')


def read_text(book_id, txt_path="data"):
    """
    Считывает текст произведения из файла, имя которого формируется как "<Book ID>.txt".
    """
    
    file_path = os.path.join(txt_path, f"{book_id}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        print(book_id)
        return f.read()

def compute_all_metrics(text):
    print("compute_all_metrics")
    """
    Вычисляет все метрики для переданного текста.
    
    Объединяет результаты функций:
      - analyze_lexical_diversity (лексическое разнообразие: TTR, MTLD, YuleK);
      - analyze_text_lexical_density (лексическая плотность);
      - analyze_lexical_complexity (средняя длина слова, среднее число слогов, доля длинных слов);
      - analyze_rare_words (доля редких слов);
      - analyze_cliches (доля клишированных выражений);
      - analyze_emotional_lexicon (частота эмоционально окрашенных слов).
    """
    metrics = {}
    metrics.update(analyze_lexical_diversity(text))
    metrics.update(analyze_text_lexical_density(text))
    metrics.update(analyze_lexical_complexity(text))
    metrics.update(analyze_rare_words(text))
    metrics.update(analyze_cliches(text))
    metrics.update(analyze_emotional_lexicon(text))
    return metrics

def main():
    # Загрузка CSV с метаданными, обязательно должна быть колонка "Book ID"
    df = pd.read_csv("./data/_merged_filtered.csv")
    
    # Считываем тексты для каждого произведения
    df["text"] = df["Book ID"].apply(read_text)
    # Отфильтровываем записи, для которых текст не найден
    df = df.dropna(subset=["text"])
    
    # Вычисляем все метрики по каждому тексту
    df["metrics"] = df["text"].apply(compute_all_metrics)
    
    # "Разворачиваем" словарь с метриками в отдельные колонки
    metrics_df = pd.concat([df.drop(columns=["metrics"]), df["metrics"].apply(pd.Series)], axis=1)
    
    # Сохраняем результат в новый CSV файл
    output_file = "data/text_metrics.csv"
    metrics_df = metrics_df.drop(columns=["text"])
    metrics_df.to_csv(output_file, index=False)
    print(f"Метрики для всех текстов сохранены в файл {output_file}")

if __name__ == '__main__':
    main()
