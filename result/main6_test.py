import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# === НАСТРОЙКИ ===
RATING_STEP = 0.5
K_FOLDS = 10
EPOCHS = 100

# === Загрузка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]
df = df[df["Rating"] <= 5]

# Преобразование рейтингов в классы
df["Rating_class"] = df["Rating"].apply(lambda x: round(x / RATING_STEP) * RATING_STEP)

print(df["Rating_class"].value_counts().sort_index())

unique_classes = sorted(df["Rating_class"].unique())
class_to_idx = {v: i for i, v in enumerate(unique_classes)}
idx_to_class = {i: v for v, i in class_to_idx.items()}
df["Class_idx"] = df["Rating_class"].map(class_to_idx)

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Class_idx"].values
num_classes = len(unique_classes)

# === PyTorch модель ===
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Кросс-валидация ===
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
accuracies = []
top2s = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # В тензоры
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Создание и обучение модели
    model = MLPClassifier(X.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Оценка
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        true = y_test_tensor.cpu().numpy()

    acc = accuracy_score(true, preds)
    top2 = top_k_accuracy_score(true, probs, k=2, labels=list(range(num_classes)))
    print(f"✅ Accuracy: {acc:.4f} | Top-2 Accuracy: {top2:.4f}")

    accuracies.append(acc)
    top2s.append(top2)

# === Финальный вывод ===
print("\n📊 Кросс-валидация завершена:")
print(f"Средняя Accuracy:     {np.mean(accuracies):.4f}")
print(f"Средняя Top-2 Accuracy: {np.mean(top2s):.4f}")
