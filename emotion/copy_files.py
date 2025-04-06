import os
import csv
import shutil

base_dir = 'russian_literature_data_fixed/prose'

# Определяем путь к папке out (она должна находиться в той же директории, что и скрипт)
script_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = os.path.join(script_dir, 'out')
os.makedirs(out_dir, exist_ok=True)

for root_main, dirs_main, files_main in os.walk(base_dir):
    for directory in dirs_main:
        subdir_path = os.path.join(base_dir, directory)
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.txt'):
                    title = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    author = os.path.basename(os.path.dirname(file_path))
                    
                    # Проверяем размер файла и копируем, если он не превышает 100 кб
                    file_size = os.path.getsize(file_path)  # размер в байтах
                    if file_size <= 10000 * 1024:
                        dest_path = os.path.join(out_dir, file)
                        shutil.copy(file_path, dest_path)
                        print(f"Файл '{file}' скопирован в папку out (размер: {file_size} байт)")
            break
