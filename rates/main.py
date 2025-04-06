import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import pandas as pd
import time

# Загрузка CSV с названиями произведений
df = pd.read_csv("results.csv")

# Подготовим столбец для результатов
ratings = []

# Заголовки для HTTP-запросов
headers = {"User-Agent": "Mozilla/5.0"}

def get_book_rating_by_title(title, max_retries=3):
    title = title.replace(" ", "+")
    encoded_title = urllib.parse.quote(title)
    search_url = f"https://www.livelib.ru/find/{encoded_title}"
    print(search_url)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for attempt in range(max_retries):
        try:
            response = session.get(search_url, timeout=10)
            if response.status_code != 200:
                return -1

            if "Пожалуйста, подождите пару секунд, идет перенаправление на сайт" in response.text:
                print("[WAIT] Обнаружена заглушка. Ждём 8 секунд и повторяем запрос...")
                time.sleep(10)
                continue  # повторим с той же сессией

            soup = BeautifulSoup(response.text, "html.parser")
            target_div = soup.find("div", class_="brow-stats-outer")
            if not target_div:
                return -1

            rating_span = target_div.find("span", title=re.compile(r"Рейтинг"))
            if not rating_span:
                return -1

            title_attr = rating_span["title"]
            match = re.search(r"Рейтинг\s+([0-9.]+)", title_attr)
            if match:
                rating_value = float(match.group(1))
                return round(rating_value, 3)
            else:
                return -1

        except Exception as e:
            print(f"[ERROR] Ошибка запроса: {e}")
            time.sleep(5)

    return -1

# Обработка всех заголовков
for index, row in df.iterrows():
    title = row["Title"]
    author = row["Author"]
    print(f"[INFO] Обрабатывается: {author} {title}")
    rating = get_book_rating_by_title(f"{author}+{title}")
    print(f"[RESULT] Рейтинг: {rating}")
    ratings.append(rating)
    time.sleep(6.5)  # Задержка между запросами

# Добавляем результат в таблицу и сохраняем
df["Rating"] = ratings
df.to_csv("results_with_ratings.csv", index=False)
print("[INFO] Обработка завершена. Файл сохранён как 'results_with_ratings.csv'")
