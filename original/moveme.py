import os
import shutil
import pandas as pd
import glob
import random

def move_texts_by_author(title_query: str, csv_path="results.csv", source_folder="./out/", target_folder="./test/"):
    os.makedirs(target_folder, exist_ok=True)

    # Загрузка данных
    results = pd.read_csv(csv_path)
    title_to_author = dict(zip(results['Title'], results['Author']))
    author_to_titles = {}

    # Группировка названий по авторам
    for title, author in title_to_author.items():
        author_to_titles.setdefault(author, []).append(title)

    # Получение целевого автора
    target_author = title_to_author.get(title_query)
    if not target_author:
        print(f"Автор для произведения \"{title_query}\" не найден.")
        return

    print(f"Выбран автор: {target_author}")
    moved = 0

    # Все .txt файлы в исходной папке
    all_txt_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(source_folder, "*.txt"))}

    # Переместить все файлы целевого автора
    for title in author_to_titles.get(target_author, []):
        if title in all_txt_files:
            src = all_txt_files[title]
            dst = os.path.join(target_folder, os.path.basename(src))
            shutil.move(src, dst)
            print(f"[Автор: {target_author}] Перемещён: {src} -> {dst}")
            moved += 1

    # Переместить по два случайных текста остальных авторов
    for other_author, titles in author_to_titles.items():
        if other_author == target_author:
            continue
        candidates = [t for t in titles if t in all_txt_files]
        selected = random.sample(candidates, min(2, len(candidates)))
        for title in selected:
            src = all_txt_files[title]
            dst = os.path.join(target_folder, os.path.basename(src))
            shutil.move(src, dst)
            print(f"[Автор: {other_author}] Перемещён: {src} -> {dst}")
            moved += 1

    print(f"\nВсего перемещено файлов: {moved}")

# Пример использования:
move_texts_by_author("Жги, еретик!")
