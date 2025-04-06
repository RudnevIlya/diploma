# %% [markdown]
# # Обучение модели для оценки метафоричности предложений с сохранением модели и поддержкой GPU
#
# В этом ноутбуке мы:
# 1. Загружаем датасет из файла `all_verbs.csv`, агрегируем данные по предложениям.
# 2. Выполняем предобработку текста.
# 3. Извлекаем признаки с помощью ruBERT и ручного признака (поиск конструкции "как + слово").
# 4. Загружаем сохранённую модель или обучаем новую, сохраняем её в файл.
# 5. Оцениваем модель и выводим Accuracy и F1 Score.
# 6. Определяем функцию `estimate_metaphoricity(text: str) -> float`, которая возвращает оценку метафоричности текста.

# %% [code]
import os
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')  # для токенизации и разбиения на предложения
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModel
import pymorphy2
import joblib

# Инициализируем анализатор для лемматизации с pymorphy2
morph = pymorphy2.MorphAnalyzer()

# %% [markdown]
# ## 1. Загрузка и агрегирование датасета
#
# Файл `all_verbs.csv` содержит:
# - **sentID**: идентификатор (или слово)
# - **sent**: предложение с примером употребления
# - **class**: разметка (0 – буквальное, 1 – метафорическое)
#
# Агрегируем данные по предложениям: если в одном предложении встречается хотя бы одна метка 1, то всё предложение считается метафоричным.

# %% [code]
# Загрузка датасета
df = pd.read_csv('all_verbs.csv', sep='\t')

# Удаление столбцов с номерами строк, если они присутствуют
for col in ['row', 'Unnamed: 0']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Обработка столбца с идентификатором слова: удаляем часть после символа '#'
if 'sentID' in df.columns:
    df['sentID'] = df['sentID'].astype(str).apply(lambda x: x.split('#')[0])
elif 'sentId' in df.columns:
    df['sentId'] = df['sentId'].astype(str).apply(lambda x: x.split('#')[0])
else:
    print("Столбец с идентификатором (sentID/sentId) не найден.")

print("Первые строки датасета после предобработки:")
print(df.head())

# Агрегация: группируем по столбцу "sent" и выбираем максимальное значение class.
df_grouped = df.groupby("sent", as_index=False).agg({"class": "max"})
print("\nКоличество уникальных предложений:", df_grouped.shape[0])
print(df_grouped.head())

# %% [markdown]
# ## 2. Предобработка текста
#
# Приводим текст к нижнему регистру, токенизируем и лемматизируем с помощью pymorphy2.

# %% [code]
def preprocess_text(text: str):
    text = text.lower().strip()  # приведение к нижнему регистру и удаление лишних пробелов
    tokens = nltk.word_tokenize(text, language='russian')  # токенизация
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]  # лемматизация
    return lemmas

# Пример предобработки
example = "Ее глаза сияли, как звезды на ночном небе."
print("Лемматизированный текст:", preprocess_text(example))

# %% [markdown]
# ## 3. Извлечение признаков
#
# Для каждого предложения извлекаем:
# 1. Контекстный эмбеддинг с помощью ruBERT (используем эмбеддинг [CLS]).
# 2. Ручной бинарный признак: наличие конструкции "как + слово".

# %% [code]
# Загружаем ruBERT
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Поддержка вычислений на GPU: если доступна, перемещаем модель на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

def extract_features(sentence: str):
    # 1. Извлечение эмбеддинга с помощью ruBERT
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Берем эмбеддинг первого токена ([CLS])
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # размерность ~768

    # 2. Ручной признак: поиск конструкции "как + слово"
    pattern = r"как\s+\w+"
    manual_feature = 1 if re.search(pattern, sentence.lower()) else 0

    # Объединяем признаки: конкатенация эмбеддинга и бинарного признака
    features = np.concatenate([embedding, np.array([manual_feature])])
    return features

# Пример извлечения признаков
feat = extract_features(example)
print("Размер вектора признаков:", feat.shape)

# %% [markdown]
# ## 4. Подготовка данных и обучение/загрузка классификатора
#
# Извлекаем признаки для всех предложений, разделяем данные на обучающую и тестовую выборки.
# Если файл с моделью существует, загружаем её, иначе обучаем логистическую регрессию и сохраняем.

# %% [code]
# Извлечение признаков для каждого предложения
X = np.array([extract_features(sent) for sent in df_grouped["sent"]])
y = df_grouped["class"].values

print("Размер X:", X.shape)
print("Размер y:", y.shape)

# Разделяем данные: 80% для обучения, 20% для теста
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Путь для сохранения модели
model_file = 'logreg_model.pkl'

if os.path.exists(model_file):
    print("Загрузка модели из файла:", model_file)
    clf = joblib.load(model_file)
else:
    print("Обучение модели...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_file)
    print("Модель сохранена в файл:", model_file)

# %% [markdown]
# ## 5. Оценка модели
#
# Вычисляем Accuracy и F1 Score на тестовой выборке.

# %% [code]
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", acc)
print("F1 Score:", f1)

# %% [markdown]
# ## 6. Функция оценки метафоричности текста
#
# Функция `estimate_metaphoricity(text: str) -> float`:
# - Делит входной текст на предложения.
# - Для каждого предложения извлекает признаки и получает вероятность метафоричности (вероятность класса 1).
# - Возвращает среднее значение вероятностей по всем предложениям.

# %% [code]
def estimate_metaphoricity(text: str) -> float:
    sentences = nltk.sent_tokenize(text, language='russian')
    if not sentences:
        return 0.0
    metaphor_probs = []
    for sent in sentences:
        features = extract_features(sent).reshape(1, -1)
        prob = clf.predict_proba(features)[0, 1]
        metaphor_probs.append(prob)
        print(sent, features, prob)
    return float(np.mean(metaphor_probs))

# %% [markdown]
# ## 7. Тестирование функции на примере
#
# Пример текста с несколькими предложениями.

# %% [code]
with open("test.txt", 'r', encoding='utf-8') as txt_file:
        sample_text = txt_file.read()
print("Оценка метафоричности текста:", estimate_metaphoricity(sample_text))
