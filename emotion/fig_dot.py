import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Считываем CSV файл
df = pd.read_csv('results.csv', encoding='utf-8')
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Фильтруем данные: оставляем только записи, где Length < 500000
df_filtered = df[df['Length'] > 1000]
df_filtered = df_filtered[df_filtered['Author'] != "Morningstar"]

# Получаем список уникальных авторов
authors = df_filtered['Author'].unique()
num_authors = len(authors)

# Получаем colormap с нужным количеством уникальных цветов (например, tab20)
cmap = cm.get_cmap('tab20', num_authors)

plt.figure(figsize=(10, 6))

# Строим scatter plot для каждого автора, назначая уникальный цвет из cmap
for i, author in enumerate(authors):
    subset = df_filtered[df_filtered['Author'] == author]
    plt.scatter(subset['Length'], subset['Value'], 
                color=cmap(i), alpha=1, label=author)

plt.xlabel('Количество символов')
plt.ylabel('Средняя интенсивность')
plt.title('Зависимость Эмоциональности от Длины текста')
plt.xscale('log')  # логарифмическая шкала для оси X
plt.grid(True)
plt.legend(title="Author")
plt.tight_layout()
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

# # Считываем CSV файл
# df = pd.read_csv('results.csv', encoding='utf-8')
# df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
# df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# # Фильтруем данные: оставляем только записи, где Length > 1000 и Author не равен "Morningstar"
# df_filtered = df[df['Length'] > 1000]
# df_filtered = df_filtered[df_filtered['Author'] != "Morningstar"]

# # Получаем список уникальных авторов
# authors = df_filtered['Author'].unique()
# num_authors = len(authors)

# # Получаем colormap с нужным количеством уникальных цветов (например, tab20)
# cmap = cm.get_cmap('tab20', num_authors)

# plt.figure(figsize=(10, 6))

# # Строим scatter plot для каждого автора, назначая уникальный цвет из cmap
# # Теперь x - Value, а y - Length
# for i, author in enumerate(authors):
#     subset = df_filtered[df_filtered['Author'] == author]
#     plt.scatter(subset['Value'], subset['Length'], 
#                 color=cmap(i), alpha=1, label=author)

# plt.xlabel('Value (средняя интенсивность)')
# plt.ylabel('Length (количество символов)')
# plt.title('Зависимость Length от Value')
# plt.yscale('log')  # Логарифмическая шкала для оси Y, поскольку теперь y - Length
# plt.grid(True)
# plt.legend(title="Author")
# plt.tight_layout()
# plt.show()
