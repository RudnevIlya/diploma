import zipfile
import os

def fix_zip_encoding(zip_path, extract_to, from_encoding='cp437', to_encoding='utf-8'):
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            # Декодируем имя файла вручную
            name_bytes = info.filename.encode('cp437')  # то, как zip думает, что это байты
            fixed_name = name_bytes.decode(to_encoding)  # реальная раскодировка

            # Путь для распаковки
            target_path = os.path.join(extract_to, fixed_name)

            # Создаём папки
            if info.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with z.open(info) as source, open(target_path, 'wb') as target:
                    target.write(source.read())

    print(f'Успешно извлечено в {extract_to} с исправленными именами файлов')

# Пример использования:
zip_path = 'archive.zip'
extract_to = 'russian_literature_data_fixed'
fix_zip_encoding(zip_path, extract_to)
