import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# === Настройка модели BERT ===
bert_model_name = "sberbank-ai/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Загрузка таблицы с метаданными ===
df = pd.read_csv("with_readability_manual.csv")
bert_embeddings = {}

def get_bert_embedding(text, method="cls"):
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        if method == "cls":
            return last_hidden[:, 0, :].squeeze().cpu().numpy()  # CLS-токен
        elif method == "mean":
            return last_hidden.mean(dim=1).squeeze().cpu().numpy()  # усреднение
        else:
            raise ValueError("method must be 'cls' or 'mean'")

# === Чтение текстов из папки ===
for filename in tqdm(os.listdir("out")):
    if filename.endswith(".txt"):
        title = filename.replace(".txt", "").strip()
        try:
            with open(os.path.join("out", filename), "r", encoding="utf-8") as f:
                text = f.read()
                emb = get_bert_embedding(text, method="cls")
                bert_embeddings[title] = emb
        except Exception as e:
            print(f"⚠️ Ошибка в {filename}: {e}")

# === Вектор → DataFrame ===
bert_df = pd.DataFrame.from_dict(bert_embeddings, orient="index")
bert_df.index.name = "Title"
bert_df.reset_index(inplace=True)

# === Объединение с основной таблицей ===
df["Title"] = df["Title"].str.strip()
merged = df.merge(bert_df, on="Title", how="left")

# === Сохранение ===
merged.to_csv("merged_with_bert.csv", index=False)
print("✅ Сохранено: merged_with_bert.csv (признаки + BERT)")
