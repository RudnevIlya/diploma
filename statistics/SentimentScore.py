import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# === Загрузка модели ===
model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Использование GPU если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Подготовка данных ===
df = pd.read_csv("with_readability_manual.csv")
sentiments = {}

def compute_sentiment(text):
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    sentiment_score = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]
    return round(sentiment_score, 4)

# === Обработка текстов ===
for filename in tqdm(os.listdir("out")):
    if filename.endswith(".txt"):
        title = filename.replace(".txt", "").strip()
        try:
            with open(os.path.join("out", filename), "r", encoding="utf-8") as f:
                text = f.read()
                score = compute_sentiment(text)
                sentiments[title] = score
        except Exception as e:
            print(f"⚠️ Ошибка в файле {filename}: {e}")

# === Объединение с таблицей ===
df["SentimentScore"] = df["Title"].apply(lambda t: sentiments.get(t.strip(), None))
df.to_csv("merged_results_with_sentiment.csv", index=False)
print("✅ Готово! Сохранено в merged_results_with_sentiment.csv")
