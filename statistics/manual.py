import os
import pandas as pd
from tqdm import tqdm
import re

# === Настройки ===
CSV_PATH = "merged_results.csv"
TXT_DIR = "out"
OUTPUT_CSV = "with_readability_manual.csv"

# === Загрузка CSV ===
df = pd.read_csv(CSV_PATH)
df["Title_clean"] = df["Title"].astype(str).str.strip().str.replace(r'[\\/*?:"<>|]', '', regex=True)

# === Инициализация новых колонок ===
df["AvgWordLength"] = None
df["AvgSentenceLength"] = None
df["LongWordRatio"] = None
df["LexicalDiversity"] = None

def compute_manual_readability(text: str):
    # Очистка и токенизация
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{2,}\b', text)
    num_words = len(words)
    num_sentences = len(sentences)

    avg_word_len = sum(len(w) for w in words) / num_words if num_words > 0 else 0
    avg_sentence_len = num_words / num_sentences if num_sentences > 0 else 0
    long_words = [w for w in words if len(w) > 7]
    long_word_ratio = len(long_words) / num_words if num_words > 0 else 0
    lexical_diversity = len(set(words)) / num_words if num_words > 0 else 0

    return {
        "AvgWordLength": round(avg_word_len, 3),
        "AvgSentenceLength": round(avg_sentence_len, 3),
        "LongWordRatio": round(long_word_ratio, 3),
        "LexicalDiversity": round(lexical_diversity, 3)
    }

# === Обработка файлов ===
for filename in tqdm(os.listdir(TXT_DIR), desc="📂 Обработка файлов"):
    if filename.endswith(".txt"):
        title = os.path.splitext(filename)[0]
        filepath = os.path.join(TXT_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            scores = compute_manual_readability(text)
            idxs = df[df["Title_clean"] == title].index
            if not idxs.empty:
                for metric, val in scores.items():
                    df.loc[idxs, metric] = val
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {filename}: {e}")

df.drop(columns=["Title_clean"], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Сохранено: {OUTPUT_CSV}")
