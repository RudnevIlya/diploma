import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import csv

# ==============================
# Настройки
# ==============================
ENABLE_LEMMATIZATION = True           # Если False, лемматизация не применяется
BATCH_SIZE = 16                       # Размер батча для инференса
MODEL_SAVE_PATH = "./"                # Путь для сохранения/загрузки модели

base_dir = 'russian_literature_data_fixed/prose'
csv_file = 'results.csv'
csv_headers = ['Author', 'Title', 'Value', 'Length']

# Пороговые значения для итоговой классификации эмоциональности
THRESHOLD_LOW = 0.33
THRESHOLD_HIGH = 0.67

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используем устройство:", device)

# ==============================
# Опциональная предобработка с Natasha
# ==============================
if ENABLE_LEMMATIZATION:
    from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
    segmenter = Segmenter()
    emb_natasha = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb_natasha)
    morph_vocab = MorphVocab()

    def preprocess(text):
        """
        Токенизация и лемматизация текста с помощью Natasha.
        Отбрасываются токены, у которых отсутствует лемма или она не состоит только из букв.
        """
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        tokens = [token.lemma for token in doc.tokens if token.lemma and token.lemma.isalpha()]
        return " ".join(tokens)
else:
    def preprocess(text):
        return text

# ==============================
# Загрузка или инициализация модели и токенизатора
# ==============================
if os.path.exists(MODEL_SAVE_PATH):
    print("Загружаем модель и токенизатор из файла...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
else:
    print("Загружаем модель и токенизатор из Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("sismetanin/sbert-ru-sentiment-rusentiment")
    model = AutoModelForSequenceClassification.from_pretrained("sismetanin/sbert-ru-sentiment-rusentiment")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
model.to(device)
model.eval()

print("Конфигурация модели:", model.config.id2label)
# Предполагается, что модель возвращает 5 классов: 0, 1, 2, 3, 4

# ==============================
# Функция инференса (батчами)
# ==============================
def predict_sentiment(texts, batch_size=BATCH_SIZE):
    """
    Получает список предложений и возвращает список предсказанных меток (числа от 0 до 4).
    """
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Обработка батчей"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
        predictions.extend(batch_preds)
    return predictions

# ==============================
# Функция для агрегации с учетом весов
# ==============================
weight_map = {
    0: 1,  # Очень негативная
    1: 0,  # Негативная
    2: 0,  # Нейтральная
    3: 0,  # Позитивная
    4: 1   # Очень позитивная
}

def map_intensity_to_label(avg_intensity):
    """
    Преобразует среднюю интенсивность в итоговую оценку:
      - avg < THRESHOLD_LOW -> "низкая"
      - THRESHOLD_LOW <= avg < THRESHOLD_HIGH -> "средняя"
      - avg >= THRESHOLD_HIGH -> "высокая"
    """
    if avg_intensity < THRESHOLD_LOW:
        return "низкая"
    elif avg_intensity < THRESHOLD_HIGH:
        return "средняя"
    else:
        return "высокая"

# ==============================
# Основная функция для анализа текста
# ==============================
def analyze_text(text):
    """
    Обрабатывает входной текст:
      1. Приводит текст к одной строке.
      2. Разбивает текст на предложения.
      3. Применяет предобработку и инференс модели к каждому предложению.
      4. Если предложение заканчивается на восклицательный знак, его интенсивность устанавливается равной 1.
      5. Для остальных предложений используется предсказание модели с учетом weight_map.
      6. Агрегирует интенсивность эмоций и возвращает итоговую оценку.
    """
    # Убираем переносы строки и лишние пробелы
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # Разбивка на предложения по знакам препинания . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        print("Текст не содержит предложений для анализа.")
        return None

    # Предобработка предложений
    processed_sentences = [preprocess(s) for s in sentences]
    processed_sentences = [ps if ps.strip() != "" else s for ps, s in zip(processed_sentences, sentences)]

    # Предсказания модели для каждого предложения
    preds = predict_sentiment(processed_sentences)

    intensities = []
    for s, pred in zip(sentences, preds):
        if s.strip().endswith('!'):
            intensity = 1  # В случае восклицательного предложения
        else:
            intensity = weight_map.get(pred, 0)
        intensities.append(intensity)

    # Вычисляем среднюю интенсивность
    avg_intensity = sum(intensities) / len(intensities)
    final_emotion = map_intensity_to_label(avg_intensity)

    print("Средняя интенсивность эмоций:", avg_intensity)
    print("Итоговая оценка эмоциональности текста:", final_emotion)

    return final_emotion, avg_intensity

# ==============================
# Загрузка CSV (если существует) в словарь для обновления
# Ключ – Title, значение – строка данных
# ==============================
csv_data = {}
if os.path.exists(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data[row['Title']] = row
else:
    # Создаем файл с заголовками, если его нет
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

# ==============================
# Обход файлов
# ==============================
for root_main, dirs_main, files_main in os.walk(base_dir):
    for directory in dirs_main:
        subdir_path = os.path.join(base_dir, directory)
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.txt'):
                    title = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    author = os.path.basename(os.path.dirname(file_path))
                    
                    # Считываем содержимое файла и вычисляем его длину
                    with open(file_path, 'r', encoding='utf-8') as txt_file:
                        input_text = txt_file.read()
                    file_length = len(input_text)
                    
                    # Если файл уже обработан, обновляем столбец Length и не проводим анализ
                    if title in csv_data:
                        csv_data[title]['Length'] = str(file_length)
                        print(f"Файл '{title}' найден в CSV. Обновлена длина: {file_length}")
                    else:
                        # Выполняем анализ текста
                        final_emotion, value = analyze_text(input_text)
                        csv_data[title] = {
                            'Author': author,
                            'Title': title,
                            'Value': value,
                            'Length': str(file_length)
                        }
                        print(f"Обработан файл: {file_path} | Author: {author}, Title: {title}, Value: {value}, Length: {file_length}")
                    
                    # Перезаписываем CSV файл после обработки каждого файла
                    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writeheader()
                        for row in csv_data.values():
                            writer.writerow(row)
            break
