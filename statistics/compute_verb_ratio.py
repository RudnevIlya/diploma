import os
import re
import pandas as pd
import pymorphy2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# === Модель для сентимента ===
model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Морфоанализатор ===
morph = pymorphy2.MorphAnalyzer()

# === Настройки признаков ===
ABSTRACT_SUFFIXES = ("ость", "изм", "ие", "ция", "ность", "ание", "ение")
NEGATIONS = {"не", "нет", "никто", "ничто", "нельзя", "никак", "ни"}

# === Данные ===
df = pd.read_csv("with_readability_manual.csv")
sentiments = {}
verb_ratios = {}
adj_ratios = {}
quotes_counts = {}
negation_counts = {}
excl_counts = {}
abstract_ratios = {}

def compute_sentiment(text):
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    sentiment_score = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]
    return round(sentiment_score, 4)

def compute_verb_ratio(text):
    words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
    if not words:
        return 0.0
    verb_count = sum(1 for word in words if 'VERB' in morph.parse(word)[0].tag or 'INFN' in morph.parse(word)[0].tag)
    return round(verb_count / len(words), 4)

def compute_additional_features(text):
    words = re.findall(r'\b[а-яА-ЯёЁ]{2,}\b', text.lower())
    total = len(words) or 1
    tags = [morph.parse(w)[0].tag for w in words]

    adj_ratio = sum("ADJF" in tag for tag in tags) / total
    quotes_count = len(re.findall(r"[«»\"“”']", text))
    negation_count = sum(1 for w in words if w in NEGATIONS)
    excl_count = text.count("!")
    abstract_count = sum(1 for w in words if w.endswith(ABSTRACT_SUFFIXES)) / total

    return round(adj_ratio, 4), quotes_count, negation_count, excl_count, round(abstract_count, 4)

# === Обработка текстов ===
for filename in tqdm(os.listdir("out"), desc="Обработка файлов"):
    if filename.endswith(".txt"):
        title = filename.replace(".txt", "").strip()
        try:
            with open(os.path.join("out", filename), "r", encoding="utf-8") as f:
                text = f.read()
                # sentiments[title] = compute_sentiment(text)
                # verb_ratios[title] = compute_verb_ratio(text)
                adj, quotes, neg, excl, abstr = compute_additional_features(text)
                adj_ratios[title] = adj
                quotes_counts[title] = quotes
                negation_counts[title] = neg
                excl_counts[title] = excl
                abstract_ratios[title] = abstr
        except Exception as e:
            print(f"⚠️ Ошибка в файле {filename}: {e}")

# === Объединение с таблицей ===
# df["SentimentScore"] = df["Title"].apply(lambda t: sentiments.get(t.strip(), None))
# df["VerbRatio"] = df["Title"].apply(lambda t: verb_ratios.get(t.strip(), None))
df["AdjRatio"] = df["Title"].apply(lambda t: adj_ratios.get(t.strip(), None))
df["QuotesCount"] = df["Title"].apply(lambda t: quotes_counts.get(t.strip(), None))
df["NegationCount"] = df["Title"].apply(lambda t: negation_counts.get(t.strip(), None))
df["ExclamationCount"] = df["Title"].apply(lambda t: excl_counts.get(t.strip(), None))
df["AbstractNounRatio"] = df["Title"].apply(lambda t: abstract_ratios.get(t.strip(), None))

df.to_csv("merged_results_with_all_features.csv", index=False)
print("✅ Готово! Сохранено в merged_results_with_all_features.csv")
