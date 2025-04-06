import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Задаем ширину интервала одного столбца
bin_interval = 0.01

# Считываем CSV файл
df = pd.read_csv('results.csv', encoding='utf-8')
values = df['Value']

# Определяем точки разбиения диапазона от 0 до 1 с заданным интервалом
bin_edges = np.arange(0, 1 + bin_interval, bin_interval)

# Вычисляем количество значений в каждом интервале
counts, _ = np.histogram(values, bins=bin_edges)

# Вычисляем центры интервалов для размещения столбцов
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Формируем подписи для каждого интервала вида "0.00-0.01", "0.01-0.02", и т.д.
labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]

# Построение графика
plt.figure(figsize=(20, 8))
plt.bar(bin_centers, counts, width=bin_interval, edgecolor='black')
plt.xlabel('Интервалы значений (Value)')
plt.ylabel('Количество записей')
plt.title('Гистограмма распределения значений столбца Value')

# Устанавливаем подписи по оси X для каждого столбца
plt.xticks(bin_centers, labels, rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
