import pandas as pd

# Результаты моделей (значения взяты из ваших сообщений)
results = {
    "Модель": [
        "Моя модель (рассчитанные метрики)",
        "TF-IDF (без лемматизации)",
        "TF-IDF (с лемматизацией)",
        "Baseline: статистические признаки + LR"
    ],
    "MSE": [
        9.001197691732392e-05,
        0.00012264345604408262,
        0.00022644593338387353,
        0.00012026810010638734
    ],
    "RMSE": [
        0.009487464198473896,
        0.011074450597843788,
        0.015048120593079839,
        0.010966681362490082
    ],
    "MAE": [
        0.0072151655122446,
        0.008444934951471455,
        0.011624220143983986,
        0.008351558997508833
    ],
    "Spearman ρ": [
        0.3733238834462061,
        0.2453629884556798,
        0.10937216715598799,
        0.19187740780415877
    ],
    "Kendall Tau": [
        0.29016227205057166,
        0.16681671009028623,
        0.07356933212405589,
        0.12998547897674745
    ],
    "NDCG": [
        0.9091079478353707,
        0.8439917740073049,
        0.8328213000776714,
        0.8502919511913625
    ],
    "R²": [
        0.19076212745764065,  # Значение R² только у "Моей модели"
        "—", "—", "—"
    ]
}

df_results = pd.DataFrame(results)
print(df_results)

# Для наглядного вывода можно сохранить DataFrame в CSV или отобразить в виде markdown:
print(df_results.to_markdown(index=False))
