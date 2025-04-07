import pandas as pd
from functools import reduce

import pandas as pd
from functools import reduce

import pandas as pd
from functools import reduce

def merge_csv_files(file_paths, on="Title", save_path=None):
    """
    Объединяет несколько CSV-файлов по ключу (по умолчанию Title).
    Убирает дубликаты колонок, заполняет NaN из других источников.
    """

    # Загружаем файлы
    dfs = [pd.read_csv(path, index_col=0) for path in file_paths]

    # Последовательно объединяем
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=on, how="outer", suffixes=("", "_dup")), dfs)

    # Обработка колонок *_dup
    dup_cols = [col for col in list(merged_df.columns) if col.endswith("_dup")]

    for col in dup_cols:
        base = col[:-4]

        # ⛑ Проверка на существование: пропускаем, если колонка уже удалена
        if col not in merged_df.columns:
            continue

        if base in merged_df.columns:
            base_col = merged_df[base]
            dup_col = merged_df[col]

            # Проверка на тип: если обе — Series
            if isinstance(base_col, pd.Series) and isinstance(dup_col, pd.Series):
                merged_df[base] = base_col.combine_first(dup_col)
            else:
                print(f"⚠ Пропуск конфликтной колонки: {base}")
        else:
            # Если нет основной колонки — переименовываем
            merged_df.rename(columns={col: base}, inplace=True)

        # Удаляем дубликат, если всё ещё существует
        if col in merged_df.columns:
            merged_df.drop(columns=[col], inplace=True)

    # Сохраняем
    if save_path:
        merged_df.to_csv(save_path, index=False)
        print(f"✅ Объединённый файл сохранён: {save_path}")

    return merged_df



merged = merge_csv_files(
    file_paths=[
        "merged_results_with_sentiment.csv",
        "merged_results_with_all_features.csv",
        "merged_results_with_sentiment_and_verbratio.csv",
        "merged_with_bert.csv",
        "merged_with_bert_cosine.csv"
        
    ],
    on="Title",

    save_path="final_merged_output_with_bert.csv"
)

