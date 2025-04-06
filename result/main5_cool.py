import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

RATING_STEP = 0.1

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 4]
df = df[df["Rating"] <= 5]

# Классификация с шагом 0.1
df["Rating_class"] = df["Rating"].apply(lambda x: round(x / RATING_STEP) * RATING_STEP)
unique_classes = sorted(df["Rating_class"].unique())
class_to_idx = {v: i for i, v in enumerate(unique_classes)}
idx_to_class = {i: v for v, i in class_to_idx.items()}
df["Class_idx"] = df["Rating_class"].map(class_to_idx)

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Class_idx"].values
num_classes = len(unique_classes)

# Train/test split и стандартизация
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Перевод в тензоры и на устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Модель
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLPClassifier(X.shape[1], num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Обучение
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/200 - Loss: {loss.item():.4f}")

# Предсказания
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    true = y_test_tensor.cpu().numpy()

# Метрики
acc = accuracy_score(true, preds)
top2 = top_k_accuracy_score(true, probs, k=2, labels=list(range(num_classes)))
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ Top-2 Accuracy: {top2:.4f}")
print("\n📊 Classification Report:")
print(classification_report(
    true,
    preds,
    target_names=[str(idx_to_class[i]) for i in range(num_classes)],
    labels=list(range(num_classes)),
    zero_division=0
))


# Сохранение предсказаний
df_preds = pd.DataFrame({
    "TrueRating": [idx_to_class[i] for i in true],
    "PredictedRating": [idx_to_class[i] for i in preds]
})
df_preds.to_csv("predictions.csv", index=False)
print("📁 Предсказания сохранены в predictions.csv")

# Матрица ошибок
cm = confusion_matrix(true, preds, labels=list(range(num_classes)))
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=unique_classes, yticklabels=unique_classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Rating Class)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("📊 Матрица ошибок сохранена в confusion_matrix.png")
