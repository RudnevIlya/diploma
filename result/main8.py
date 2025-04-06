import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from catboost import CatBoostClassifier

# === НАСТРОЙКИ ===
K_FOLDS = 5
EPOCHS = 100

# === Загрузка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]

# === Группировка в 3 класса ===
def categorize(r):
    if r <= 3.0:
        return 0  # низкий
    elif r <= 4.0:
        return 1  # средний
    else:
        return 2  # высокий

df["Rating_class"] = df["Rating"].apply(categorize)
class_names = ["низкий", "средний", "высокий"]

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Rating_class"].values
num_classes = 3

# === MLP модель ===
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

mlp_accuracies, mlp_top2s = [], []
cat_accuracies, cat_top2s = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # === MLP ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

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

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        mlp_preds = torch.argmax(logits, dim=1).cpu().numpy()
        mlp_probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

    acc_mlp = accuracy_score(y_test_np, mlp_preds)
    top2_mlp = top_k_accuracy_score(y_test_np, mlp_probs, k=2, labels=list(range(num_classes)))
    mlp_accuracies.append(acc_mlp)
    mlp_top2s.append(top2_mlp)

    print(f"🔹 MLP Accuracy: {acc_mlp:.4f} | Top-2: {top2_mlp:.4f}")
    print("📊 MLP classification report:")
    print(classification_report(y_test_np, mlp_preds, target_names=class_names, labels=[0, 1, 2], zero_division=0))

    # === CatBoostClassifier ===
    cat_model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_state=42, task_type="GPU" if torch.cuda.is_available() else "CPU")
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)
    cat_probs = cat_model.predict_proba(X_test)

    acc_cat = accuracy_score(y_test, cat_preds)
    top2_cat = top_k_accuracy_score(y_test, cat_probs, k=2, labels=list(range(num_classes)))
    cat_accuracies.append(acc_cat)
    cat_top2s.append(top2_cat)

    print(f"🔸 CatBoost Accuracy: {acc_cat:.4f} | Top-2: {top2_cat:.4f}")
    print("📊 CatBoost classification report:")
    print(classification_report(y_test, cat_preds, target_names=class_names, labels=[0, 1, 2], zero_division=0))

# === Итоги ===
print("\n📈 Средние метрики по фолдам:")
print(f"MLP        → Accuracy: {np.mean(mlp_accuracies):.4f} | Top-2: {np.mean(mlp_top2s):.4f}")
print(f"CatBoost   → Accuracy: {np.mean(cat_accuracies):.4f} | Top-2: {np.mean(cat_top2s):.4f}")
