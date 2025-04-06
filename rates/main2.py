import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.parse

# Загружаем CSV
df = pd.read_csv("results.csv")
ratings = []

# Настройка браузера (без GUI)
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

def get_rating_selenium(title):
    try:
        encoded = urllib.parse.quote(title.replace(" ", "+"))
        url = f"https://www.livelib.ru/find/{encoded}"
        driver.get(url)

        # Ждём максимум 15 секунд появления рейтинга
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "brow-stats-outer"))
        )

        block = driver.find_element(By.CLASS_NAME, "brow-stats-outer")
        rating_span = block.find_element(By.XPATH, './/span[@title[contains(., "Рейтинг")]]')
        title_attr = rating_span.get_attribute("title")

        match = re.search(r"Рейтинг\s+([0-9.]+)", title_attr)
        if match:
            return round(float(match.group(1)), 3)
        return -1
    except (TimeoutException, NoSuchElementException):
        print("[WARN] Рейтинг не найден или страница не загрузилась.")
        return -1
    except Exception as e:
        print(f"[ERROR] {e}")
        return -1

# Основной цикл
for index, row in df.iterrows():
    author = str(row["Author"])
    book = str(row["Title"])
    search_query = f"{author} {book}"
    print(f"[INFO] Обрабатывается: {search_query}")
    rating = get_rating_selenium(search_query)
    print(f"[RESULT] Рейтинг: {rating}")
    ratings.append(rating)
    time.sleep(4.0)

# Добавляем результат и сохраняем
df["Rating"] = ratings
df.to_csv("results_with_ratings.csv", index=False)

driver.quit()
print("[INFO] Обработка завершена, данные сохранены в 'results_with_ratings.csv'")
