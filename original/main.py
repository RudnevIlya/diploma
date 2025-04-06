# Импорт необходимых библиотек
import os
import glob
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pymorphy2
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import torch  # для проверки доступности GPU
import joblib  # для сохранения и загрузки моделей

# Скачиваем необходимые данные для nltk
nltk.download('punkt')

# Определяем устройство: 'cuda' если доступно, иначе 'cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Используем устройство: {device}")



# Инициализируем pymorphy2 для лемматизации
morph = pymorphy2.MorphAnalyzer()

# Инициализируем модель SentenceTransformer с указанием устройства (GPU если доступно)
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device=device)

# Функция предварительной обработки текста: приведение к нижнему регистру, токенизация, удаление мусора и лемматизация.
def preprocess_text(text: str) -> str:
    text = text.lower()  # переводим в нижний регистр
    tokens = word_tokenize(text)  # токенизация
    # Оставляем только кириллические символы
    tokens = [re.sub(r'[^а-яё]', '', token) for token in tokens]
    tokens = [token for token in tokens if token.strip() != '']
    # Лемматизация с помощью pymorphy2
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(lemmatized_tokens)

# Функция для вычисления среднего значения TF-IDF для текста.
def compute_avg_tfidf(text: str, vectorizer: TfidfVectorizer) -> float:
    tfidf_vector = vectorizer.transform([text])
    return tfidf_vector.data.mean() if tfidf_vector.nnz > 0 else 0.0

# Функция сохранения обученных моделей и векторизатора в файлы.
def save_models(classifier, tfidf_vectorizer, sentence_model, folder="models"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(classifier, os.path.join(folder, 'classifier_model.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(folder, 'tfidf_vectorizer.pkl'))
    sentence_model.save(os.path.join(folder, 'sentence_transformer_model'))
    print("Модели успешно сохранены.")

# Функция загрузки обученных моделей и векторизатора из файлов.
def load_models(folder="models"):
    classifier_path = os.path.join(folder, 'classifier_model.pkl')
    tfidf_path = os.path.join(folder, 'tfidf_vectorizer.pkl')
    sentence_model_path = os.path.join(folder, 'sentence_transformer_model')
    if os.path.exists(classifier_path) and os.path.exists(tfidf_path) and os.path.exists(sentence_model_path):
        classifier = joblib.load(classifier_path)
        tfidf_vectorizer = joblib.load(tfidf_path)
        sentence_model = SentenceTransformer(sentence_model_path, device=device)
        print("Модели успешно загружены.")
        return classifier, tfidf_vectorizer, sentence_model
    else:
        print("Не найдены сохраненные модели.")
        return None, None, None

# Чтение файла с метаданными (авторы, названия произведений и т.д.)
metadata = pd.read_csv('result.csv')

# Считываем все .txt файлы из текущей директории
txt_files = glob.glob("./out/*.txt")
data = []  # список для хранения информации о каждом произведении

for file in txt_files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    # Предполагается, что имя файла (без расширения) соответствует названию произведения
    title = os.path.splitext(os.path.basename(file))[0]
    # Находим запись в metadata по совпадению заголовка
    row = metadata[metadata['Title'] == title]
    if not row.empty:
        author = row.iloc[0]['Author']
        data.append({'title': title, 'author': author, 'raw_text': text})
    else:
        print(f"Метаданные для файла {file} не найдены.")

# Предобработка текстов и вычисление лексической разнообразности (TTR)
for item in data:
    item['processed_text'] = preprocess_text(item['raw_text'])
    try:
        lr = LexicalRichness(item['processed_text'])
        item['ttr'] = lr.ttr
    except Exception as e:
        item['ttr'] = 0.0
        print(f"Ошибка при вычислении TTR для {item['title']}: {e}")

# Подготавливаем корпус для векторизатора TF-IDF (используем предобработанные тексты)
corpus = [item['processed_text'] for item in data]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)

# Извлечение признаков для каждого произведения:
# 1. Эмбеддинг текста (SentenceTransformer)
# 2. Лексическое разнообразие (TTR)
# 3. Среднее значение TF-IDF
features = []
labels = []
for item in data:
    embedding = model.encode(item['processed_text'])
    avg_tfidf = compute_avg_tfidf(item['processed_text'], tfidf_vectorizer)
    ttr = item['ttr']
    # Объединяем эмбеддинг с дополнительными признаками в один вектор признаков
    feature_vector = np.concatenate([embedding, np.array([ttr, avg_tfidf])])
    features.append(feature_vector)
    labels.append(item['author'])

X = np.array(features)
y = np.array(labels)

# Разбиваем данные на обучающую и тестовую выборки (при необходимости)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем классификатор авторства (Logistic Regression)
classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
classifier.fit(X_train, y_train)
print("Классификатор обучен.")

# Сохраняем обученные модели
save_models(classifier, tfidf_vectorizer, model)

# Функция для оценки стилистической оригинальности текста.
def estimate_originality(text: str) -> float:
    """
    Оценивает стилистическую оригинальность текста.
    Аргументы:
        text (str): исходный текст произведения.
    Возвращает:
        float: числовая оценка оригинальности (чем выше значение, тем оригинальнее текст).
    """
    processed = preprocess_text(text)
    # Получаем эмбеддинг с использованием GPU (если доступно)
    emb = model.encode(processed)
    try:
        lr = LexicalRichness(processed)
        ttr_val = lr.ttr
    except Exception as e:
        ttr_val = 0.0
    avg_tfidf_val = compute_avg_tfidf(processed, tfidf_vectorizer)
    feature_vec = np.concatenate([emb, np.array([ttr_val, avg_tfidf_val])]).reshape(1, -1)
    # Получаем вероятности для каждого автора
    probs = classifier.predict_proba(feature_vec)[0]
    max_confidence = np.max(probs)
    # Чем ниже уверенность классификатора, тем выше оригинальность.
    originality_score = (1 - max_confidence) + ttr_val + avg_tfidf_val
    return originality_score

# Пример использования функции estimate_originality
if __name__ == "__main__":
    # Пробуем загрузить сохраненные модели, если они существуют
    loaded_classifier, loaded_tfidf, loaded_sentence_model = load_models()
    if loaded_classifier is not None:
        classifier = loaded_classifier
        tfidf_vectorizer = loaded_tfidf
        model = loaded_sentence_model
    # Если имеются данные, берем пример из первого файла, иначе тестовый пример.
    sample_text = data[0]['raw_text'] if data else "Пример текста для оценки оригинальности."
    score = estimate_originality(sample_text)
    print(f"Оценка оригинальности: {score:.4f}")
