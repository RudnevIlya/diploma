import pandas as pd
from pathlib import Path

# Пути к папкам
results_dir = Path("results")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Загрузка предсказаний
df_emo = pd.read_csv(results_dir / "pred_emo.csv")
df_lex = pd.read_csv(results_dir / "pred_lexical.csv")
df_syn = pd.read_csv(results_dir / "pred_structure.csv")
df_bert = pd.read_csv(results_dir / "pred_bert.csv")

# Переименование колонок
df_emo.rename(columns={"PredictedRating": "Pred_Emotion"}, inplace=True)
df_lex.rename(columns={"PredictedRating": "Pred_Lexical"}, inplace=True)
df_syn.rename(columns={"PredictedRating": "Pred_Structure"}, inplace=True)
df_bert.rename(columns={"PredictedRating": "Pred_BERT"}, inplace=True)

# Объединение по ID
df_merged = df_emo.merge(df_lex, on="ID") \
                  .merge(df_syn, on="ID") \
                  .merge(df_bert, on="ID")

# Загрузка целевых значений (Rating)
df_full = pd.read_csv(data_dir / "texts_with_features.csv")[["ID", "Rating"]]
df_final = df_merged.merge(df_full, on="ID")

# Сохранение результата
output_path = data_dir / "meta_model_input.csv"
df_final.to_csv(output_path, index=False)

print(f"[✓] Датафрейм объединён и сохранён в: {output_path}")
