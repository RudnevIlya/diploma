import os
import pandas as pd

# Паттерн для поиска нужных файлов
pattern = "syntactic_text_metrics"
output_filename = "syntactic_all_metrics.csv"

# Получаем список всех подходящих файлов
csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and pattern in f]

# Читаем и объединяем все CSV
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Сохраняем результат
combined_df.to_csv(output_filename, index=False)

print(f"Объединено {len(csv_files)} файлов. Результат сохранён в {output_filename}")
