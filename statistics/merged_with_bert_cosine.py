import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Загрузка данных ===
df = pd.read_csv("merged_with_bert.csv")
df = df[df["Rating"] >= 1]

# === Определение классов ===
class_ranges = {
    "низкий": (0, 3.9),
    "средний": (3.9, 4.3),
    "высокий": (4.3, 5.01)
}

def rating_to_class(rating):
    for idx, (label, (low, high)) in enumerate(class_ranges.items()):
        if low <= rating < high:
            return idx
    return len(class_ranges) - 1

df["Class"] = df["Rating"].apply(rating_to_class)

# === Выделение BERT признаков ===
bert_features = [col for col in df.columns if col.isdigit()]
X_bert = df[bert_features].values

# === Вычисление якорей (средние эмбеддинги по классам) ===
anchor_vectors = {}
for class_id in [0, 1, 2]:
    anchor_vectors[class_id] = X_bert[df["Class"] == class_id].mean(axis=0)

# === Вычисление косинусных сходств ===
def compute_similarities(vector):
    sims = []
    for class_id in [0, 1, 2]:
        sim = cosine_similarity([vector], [anchor_vectors[class_id]])[0][0]
        sims.append(sim)
    return sims

# === Добавление новых признаков ===
sims = np.array([compute_similarities(v) for v in X_bert])
df["SimLow"] = sims[:, 0]
df["SimMid"] = sims[:, 1]
df["SimHigh"] = sims[:, 2]

# === Сохранение ===
df.to_csv("merged_with_bert_cosine.csv", index=False)
print("✅ Готово! Сохранено в merged_with_bert_cosine.csv")
