import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# === Загрузка и подготовка данных ===
df = pd.read_csv("merged_results.csv")
df = df[df["Rating"] >= 0]

# Классы по границам
low, high = 3.89, 4.25
def to_class(r): return 0 if r <= low else 1 if r <= high else 2
df["Class"] = df["Rating"].apply(to_class)

features = ["HD_D", "Average_TFIDF", "Value", "Length"]
X = df[features]
y = df["Class"]

# === Обучение CatBoost (на всех данных) ===
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=11,
    random_strength=10,
    bootstrap_type="MVS",
    class_weights=[4.0, len(df)/df["Class"].value_counts()[1], 4.0],
    verbose=0
)
model.fit(X, y)

# === SHAP-анализ ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === Визуализация (суммарная важность признаков) ===
plt.title("SHAP Feature Importance")
shap.summary_plot(shap_values, X, plot_type="bar", class_names=["низкий", "средний", "высокий"])
plt.savefig("shap_summary_bar.png")

# === Визуализация распределения влияния признаков (все классы) ===
shap.summary_plot(shap_values, X, class_names=["низкий", "средний", "высокий"])

# (по желанию) индивидуальные объяснения:
# shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0], matplotlib=True)
