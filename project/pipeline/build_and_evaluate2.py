import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Добавляем базовую директорию и папку с моделями в sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
sys.path.insert(0, MODELS_DIR)

# Импортируем функции для k‑fold stacking из модуля stacking_meta_model
from stacking_meta_model import get_oof_predictions, train_final_meta_model

# Также импортируем базовые модели для формирования hold-out meta‑input
from emo_model import train_and_predict as train_and_predict_emo
from lexical_model import train_and_predict as train_and_predict_lexical
from syntax_model import train_and_predict as train_and_predict_syntax
from bert_model import train_and_predict as train_and_predict_bert

def cross_validate_meta_model(meta_df, k=5):
    """
    Выполняет k‑fold кросс‑валидацию финальной мета‑модели на объединённом meta‑input.
    Для каждого фолда обучается финальная модель на тренировочной части и оценивается на валидационной,
    затем выводятся средние метрики.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_list = []
    rmse_list = []
    r2_list = []
    
    # Для каждого фолда обучаем мета‑модель
    for train_idx, valid_idx in kf.split(meta_df):
        meta_train_fold = meta_df.iloc[train_idx].reset_index(drop=True)
        meta_valid_fold = meta_df.iloc[valid_idx].reset_index(drop=True)
        
        # Обучаем мета‑модель на фолде без дополнительных графиков
        features = ['Pred_Emotion', 'Pred_Lexical', 'Pred_Structure', 'Pred_BERT']
        X_train = meta_train_fold[features]
        y_train = meta_train_fold['Rating']
        X_valid = meta_valid_fold[features]
        y_valid = meta_valid_fold['Rating']
        
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(task_type='GPU', l2_leaf_reg=10, random_seed=42, verbose=0)
        except Exception as e:
            model = CatBoostRegressor(task_type='CPU', l2_leaf_reg=10, random_seed=42, verbose=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        
        mae = mean_absolute_error(y_valid, y_pred)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        r2 = r2_score(y_valid, y_pred)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
    
    print("\n[Meta-Model CV] Средние метрики по {} фолдам:".format(k))
    print("MAE: {:.4f} ± {:.4f}".format(np.mean(mae_list), np.std(mae_list)))
    print("RMSE: {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
    print("R²: {:.4f} ± {:.4f}".format(np.mean(r2_list), np.std(r2_list)))

def main():
    # 1. Загружаем исходный датасет
    data_path = os.path.join(BASE_DIR, 'data', 'texts_with_features.csv')
    df = pd.read_csv(data_path)
    
    # Применяем фильтрацию по Rating (если требуется)
    df = df[df["Rating"] >= 3.5]
    df = df[df["Rating"] <= 4.98]
    
    print("Исходный датасет:", df.shape)
    
    # 2. Делим данные на основную тренировочную выборку и hold-out (например, 80/20)
    df_train_full, df_holdout = train_test_split(df, test_size=0.2, random_state=42)
    print("Train_full:", df_train_full.shape, "Holdout:", df_holdout.shape)
    
    # 3. На тренировочной выборке выполняем k‑fold stacking для базовых моделей
    # Получаем out-of-fold предсказания для мета-уровня (meta_train)
    meta_train = get_oof_predictions(df_train_full, k=5)
    
    # 4. Для hold-out выборки обучаем базовые модели на полном тренировочном наборе
    # и получаем их предсказания для формирования meta_hold
    emo_hold = train_and_predict_emo(df_train_full, df_holdout)
    lexical_hold = train_and_predict_lexical(df_train_full, df_holdout)
    syntax_hold = train_and_predict_syntax(df_train_full, df_holdout)
    bert_hold = train_and_predict_bert(df_train_full, df_holdout)
    
    # Переименовываем столбцы предсказаний для корректного объединения
    emo_hold.rename(columns={'PredictedRating': 'Pred_Emotion'}, inplace=True)
    lexical_hold.rename(columns={'PredictedRating': 'Pred_Lexical'}, inplace=True)
    syntax_hold.rename(columns={'PredictedRating': 'Pred_Structure'}, inplace=True)
    bert_hold.rename(columns={'PredictedRating': 'Pred_BERT'}, inplace=True)
    
    # Объединяем предсказания базовых моделей по ID и добавляем истинный Rating из hold-out
    meta_hold = emo_hold.merge(lexical_hold, on='ID') \
                        .merge(syntax_hold, on='ID') \
                        .merge(bert_hold, on='ID')
    meta_hold = meta_hold.merge(df_holdout[['ID', 'Rating']], on='ID')
    
    # Сохраняем входные данные для мета-модели (если нужно)
    meta_input_path = os.path.join(BASE_DIR, 'data', 'meta_model_input.csv')
    meta_hold.to_csv(meta_input_path, index=False)
    print("Meta-model input сохранён в:", meta_input_path)
    
    # 5. Обучаем финальную мета-модель с использованием k‑fold stacking:
    # Функция train_final_meta_model обучит финальную модель (на meta_train) и оценит её на meta_hold.
    meta_model, meta_pred, meta_ids = train_final_meta_model(meta_train, meta_hold)
    
    # 6. Проводим k‑fold кросс‑валидацию для финальной мета‑модели и выводим средние метрики
    cross_validate = True
    if cross_validate:
        cross_validate_meta_model(meta_hold, k=5)

if __name__ == '__main__':
    main()
