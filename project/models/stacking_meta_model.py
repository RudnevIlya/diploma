import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Импорт базовых моделей (функции train_and_predict, принимающие два аргумента)
from emo_model import train_and_predict as train_and_predict_emo
from lexical_model import train_and_predict as train_and_predict_lexical
from syntax_model import train_and_predict as train_and_predict_syntax
from bert_model import train_and_predict as train_and_predict_bert

def get_oof_predictions(df, k=5):
    """
    Выполняет k‑fold stacking для базовых моделей на тренировочной выборке.
    Для каждого фолда базовые модели обучаются на части данных и предсказывают на валидационном фолде.
    Возвращает DataFrame с колонками:
      - ID
      - Pred_Emotion
      - Pred_Lexical
      - Pred_Structure
      - Pred_BERT
      - Rating  (истинное значение)
    """
    # Создаем DataFrame для хранения out‑of‑fold предсказаний, сохраняя индексы исходного df
    oof_preds = pd.DataFrame(index=df.index)
    oof_preds['ID'] = df['ID']
    oof_preds['Pred_Emotion'] = np.nan
    oof_preds['Pred_Lexical'] = np.nan
    oof_preds['Pred_Structure'] = np.nan
    oof_preds['Pred_BERT'] = np.nan

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    for train_idx, valid_idx in kf.split(df):
        print(f"Fold {fold}/{k}: Train {len(train_idx)}, Valid {len(valid_idx)}")
        df_train_fold = df.iloc[train_idx]
        df_valid_fold = df.iloc[valid_idx]
        
        # Получаем соответствующие индексы в исходном df
        valid_index_labels = df.index[valid_idx]

        # Получаем предсказания базовых моделей на валидационном фолде
        emo_pred = train_and_predict_emo(df_train_fold, df_valid_fold)
        lexical_pred = train_and_predict_lexical(df_train_fold, df_valid_fold)
        syntax_pred = train_and_predict_syntax(df_train_fold, df_valid_fold)
        bert_pred = train_and_predict_bert(df_train_fold, df_valid_fold)

        # Присваиваем значения по соответствующим индексам
        oof_preds.loc[valid_index_labels, 'Pred_Emotion'] = emo_pred['PredictedRating'].values
        oof_preds.loc[valid_index_labels, 'Pred_Lexical'] = lexical_pred['PredictedRating'].values
        oof_preds.loc[valid_index_labels, 'Pred_Structure'] = syntax_pred['PredictedRating'].values
        oof_preds.loc[valid_index_labels, 'Pred_BERT'] = bert_pred['PredictedRating'].values

        fold += 1

    # Добавляем истинные значения Rating из исходного df
    oof_preds = oof_preds.merge(df[['ID', 'Rating']], on='ID')
    return oof_preds

def train_final_meta_model(meta_train, meta_test):
    """
    Обучает финальную мета-модель (CatBoostRegressor с l2_leaf_reg=10) на мета-признаках,
    полученных от базовых моделей, и оценивает её на hold-out выборке.

    Параметры:
      meta_train: DataFrame с колонками:
         - ID, Pred_Emotion, Pred_Lexical, Pred_Structure, Pred_BERT, Rating
      meta_test: DataFrame с аналогичной структурой

    Возвращает:
      meta_model, y_pred (предсказания финальной модели на meta_test) и IDs meta_test.
    """
    features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_Structure', 'Pred_BERT']
    X_train = meta_train[features]
    y_train = meta_train['Rating']
    X_test = meta_test[features]
    y_test = meta_test['Rating']
    ids_test = meta_test['ID']
    
    try:
        meta_model = CatBoostRegressor(task_type='GPU', l2_leaf_reg=10, random_seed=42, verbose=0)
        print("Используется GPU для финальной мета-модели.")
    except Exception as e:
        print("GPU недоступен или произошла ошибка, переключение на CPU. Подробности:", e)
        meta_model = CatBoostRegressor(task_type='CPU', l2_leaf_reg=10, random_seed=42, verbose=0)
    
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("Meta-Model Metrics on Hold-out:")
    print(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    # Scatter plot: True vs Predicted
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Meta-Model: Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    scatter_path = os.path.join('results', 'meta_model_scatter.png')
    os.makedirs('results', exist_ok=True)
    plt.savefig(scatter_path)
    plt.show()
    print("Scatter plot сохранён:", scatter_path)
    
    # Bar chart: Feature importance (если доступно)
    try:
        importances = meta_model.get_feature_importance()
        plt.figure(figsize=(8,6))
        plt.bar(features, importances)
        plt.xlabel("Meta Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance of Meta-Model")
        bar_path = os.path.join('results', 'meta_model_feature_importance.png')
        plt.savefig(bar_path)
        plt.show()
        print("Bar chart важности признаков сохранён:", bar_path)
    except Exception as e:
        print("Не удалось получить важность признаков:", e)
    
    meta_result = pd.DataFrame({'ID': ids_test, 'PredictedRating': y_pred})
    final_path = os.path.join('results', 'final_prediction.csv')
    meta_result.to_csv(final_path, index=False)
    print("Итоговые предсказания сохранены в:", final_path)
    
    return meta_model, y_pred, ids_test

if __name__ == '__main__':
    # Пример использования:
    # Ожидается, что файл 'meta_model_input.csv' содержит колонки:
    # ID, Pred_Emotion, Pred_Lexical, Pred_Structure, Pred_BERT, Rating
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'meta_model_input.csv')
    df = pd.read_csv(data_path)
    
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    
    meta_model, y_pred, ids = train_final_meta_model(df_train, df_test)
    print(meta_model)
