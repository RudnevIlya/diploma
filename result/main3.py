import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# === Загрузка и фильтрация данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]

# Только числовые признаки (без AuthorRus)
features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features].values
y = df["Rating"].values

# === Разделение ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Стандартизация только для нейросети ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === PyTorch MLP на GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ PyTorch device: {device}")

class RatingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

mlp_model = RatingMLP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-4)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Обучение
for epoch in range(300):
    mlp_model.train()
    optimizer.zero_grad()
    output = mlp_model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"[MLP] Epoch {epoch+1}/300 - Loss: {loss.item():.4f}")

mlp_model.eval()
with torch.no_grad():
    y_pred_mlp = mlp_model(X_test_tensor).cpu().numpy().flatten()

# === XGBoost (GPU) ===
xgb_model = XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# === LightGBM (GPU) ===
lgb_model = LGBMRegressor(device='gpu', n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

# === CatBoost (GPU) ===
cat_model = CatBoostRegressor(task_type="GPU", devices='0', iterations=300, verbose=50, random_state=42)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)

# === Сравнение результатов ===
models = ["Neural Net (MLP)", "XGBoost", "LightGBM", "CatBoost"]
maes = [
    mean_absolute_error(y_test, y_pred_mlp),
    mean_absolute_error(y_test, y_pred_xgb),
    mean_absolute_error(y_test, y_pred_lgb),
    mean_absolute_error(y_test, y_pred_cat)
]
r2s = [
    r2_score(y_test, y_pred_mlp),
    r2_score(y_test, y_pred_xgb),
    r2_score(y_test, y_pred_lgb),
    r2_score(y_test, y_pred_cat)
]

print("\n📊 Сравнение моделей:")
for name, mae, r2 in zip(models, maes, r2s):
    print(f"{name:18} | MAE: {mae:.4f} | R²: {r2:.4f}")
