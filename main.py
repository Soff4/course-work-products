# Імпортуємо такі бібліотеки:
# - pandas (під псевдонімом pd): Використовується для обробки і аналізу даних, зазвичай в табличній формі.
# - numpy (під псевдонімом np): Використовується для роботи з числовими масивами та матрицями у Python.
# - seaborn (під псевдонімом sns): Це бібліотека для статистичної візуалізації даних, яка працює поверх бібліотеки Matplotlib.
# - matplotlib.pyplot (під псевдонімом plt): Бібліотека для створення графіків та візуалізації даних у Python.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(path):
    # Цей рядок завантажує дані з CSV-файлу за вказаним шляхом і встановлює колонку "productname" як індекс (індексуючи дані за назвою продукту).
    df_buffer = pd.read_csv(path, index_col='productname')
    # Ця операція використовує метод map для виконання функції lambda, яка видаляє символи '%' і '$' з кожного значення в DataFrame df_buffer. 
    df_buffer = df_buffer.map(lambda x: x.strip("%$"))
    # Повертає змінну df_buffer, яка містить змінені дані з CSV-файлу.
    return df_buffer

def rework_data(df):
    # Перевіряє, чи існує стовпець з назвою 'averagespread' у DataFrame self.df.
    if 'averagespread' in df.columns:
        # Якщо стовпець 'averagespread' існує, цей рядок замінює коми у числових значеннях стовпця 'averagespread' на порожній рядок. 
        # Це припускає, що значення у стовпці 'averagespread' можуть містити коми як роздільники тисяч, які потрібно видалити, щоб правильно інтерпретувати числові дані.
        df['averagespread'] = df['averagespread'].replace(',', '', regex=True)

    # Замінює всі порожні рядки у DataFrame на значення NaN (Not a Number).
    df.replace('', np.nan, inplace=True)

    # Визначає список стовпців, починаючи з другого стовпця, для подальшого перетворення на числовий тип.
    cols_to_convert = df.columns[1:7]
    # Конвертує значення у відповідних стовпцях DataFrame self.df в числовий тип даних (float). 
    # Це припускає, що всі значення у цих стовпцях мають бути числовими, а метод astype(float) спробує конвертувати їх у відповідний числовий формат.
    df[cols_to_convert] = df[cols_to_convert].astype(float)
    df.loc[:, 'productname'] = df.index
    
    return df

def clean_data(df):
    if df.isnull().values.any():
        # Пробує замінити всі пропущені значення у DataFrame на 0.0. 
        # Однак без параметру inplace=True ця зміна не буде відображена у вихідних даних; можливо, варто додати inplace=True, щоб зберегти ці зміни в DataFrame self.df.
        df.fillna(0.0, inplace=True)

    return df

def output_data(df, num_rows):
    # Вибирає перші num_rows рядків з DataFrame self.df та вибирає перші num_rows рядків з DataFrame self.df.
    # В даному випадку відбувається одразу виведення вмісту (перших num_rows рядків) за допомогою функції print(). 
    # Функція print() виводить результат методу self.df.head(num_rows) у консоль або інше місце виведення.
     return print(df.head(num_rows))



# 1
# Створюємо зміну, яка буде носити шлях до нашого набору даних
filepath = 'csv_data/ProductPriceIndex.csv'

dataframe = read_data(path=filepath)

# За допомогою функції output_data виводимо перші 15 рядків
output_data(dataframe, 15)


# 2
# Для подальшої роботи з даними із набору, нам треба конвертувати певні стовпчики в інший тип даних, а також нам треба прибрати всі зайві знаки х набору.
# В цьому нам допоможе метод rework_data()
reworked_data = rework_data(df=dataframe)

# За допомогою окремого методу, ми отримуємо статус про пусті рядки в наборі і виводимо цей статус.
# Як ми бачимо, в стовпчиках farmprice, atlantaretail та newyorkretail є пусті значення. 
# Це стає нам перешкодою у створенні певної статистики, побудові графіків і в подальших маніпуляціях.
status = reworked_data.isnull().sum()
print(f"До аналізу даних і заміни усіх NaN з допомогою fillna() / bfill(): \n{status}")

# Для виправлення ситуації ми задіємо метод clean_data(), який отримує статус (чи існують пусті рядки, чи ні) і заповнює їх за допомогою fillna().
cleaned_data = clean_data(df=reworked_data)

# Знову отримуємо статус нашого набору даних і бачимо, що більше пустих значень не існує. 
status = cleaned_data.isnull().sum()
print(f"\n\nПісля аналізу даних і заміни усіх NaN з допомогою fillna(): \n{status}")


# 3
dataframe = read_data(path=filepath)
rew_data = rework_data(df=dataframe)
cln_data = clean_data(df=rew_data)

print(f"Набір даних:\n\n")
output_data(cln_data, 15)


# 4
dataframe = cln_data
select_df = dataframe.iloc[:10, 2:6]

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: green' if v else '' for v in is_max]

def highlight_min(s):  
    is_min = s == s.min()
    return ['background-color: red' if v else '' for v in is_min]

styled_df = select_df.style.apply(highlight_max).apply(highlight_min)

styled_df.format({'atlantaretail': '{:.2f}', 'chicagoretail': '{:.2f}', 'losangelesretail': '{:.2f}',  'newyorkretail': '{:.2f}'})

styled_df


# зміна 1 для гілки 1
# 4
selected_data = dataframe.iloc[:, 2:6]

mean_data = selected_data.mean()
std_data = selected_data.std()
variance_data = np.var(selected_data)
standardized_data = (selected_data - mean_data) / std_data

print(f"Математичне сподівання для вибраних даних:\n{mean_data}")
print(f"\nДисперсія для вибраних даних:\n{variance_data}")
print(f"\nСтандартне відхилення для вибраних даних:\n{std_data}")
print(f"\nСтандартизовані вибрані дані:\n{standardized_data.iloc[:30]}")


# 5
plt.figure(figsize=(10, 6))
data = dataframe.iloc[:100, :]

sns.lineplot(data=data, x="date", y="averagespread", marker="o")

plt.xlabel('Date')
plt.ylabel('Average Spread, %')
plt.title('Average Spread between Farm and Retail Prices by Date')
plt.show()


# 5
plt.figure(figsize=(10, 6))
data = dataframe.iloc[:50, :]

sns.barplot(data=data, x='productname', y='farmprice')

plt.xlabel('Product Name')
plt.ylabel('Farm Price, $')
plt.xticks(rotation=90)
plt.title('All products Farm Price by Product Name')
plt.show()


# 5
plt.figure(figsize=(10, 6))
data = dataframe.iloc[:100, :]
sns.histplot(data=data, x='averagespread', bins=20, kde=True)

plt.xlabel('Average Spread')
plt.title('Average Spread, %')
plt.show()


# 5
df = dataframe.iloc[:, 2:6].corr()
sns.heatmap(data=df, cmap='rocket', annot=True, linewidths=.2)
plt.xlabel('Cities')
plt.ylabel('Products, $')
plt.xticks(rotation=90)
plt.title('Retail Prices in Different Cities')
plt.show()



# 6
fig, ax = plt.subplots()
data = dataframe.head(300)

for column in ['farmprice','atlantaretail','chicagoretail','losangelesretail']:
    filtered_data = data[data['date'] == '2019-05-19']
    ax.plot(filtered_data['productname'], filtered_data[column], label=column)

ax.legend()

y_values = data[data.columns[2:7]].mean().values
x_values = data['productname'].unique()

for x,y in zip(x_values, y_values):  
    ax.annotate(f"{y:.2f}", xy=(x,y), xytext=(5, 5), textcoords='offset points') 
        
ax.autoscale()
plt.xticks(rotation=90)
plt.show()


# 6
# Показує кореляції між різними параметрами.
data = dataframe.iloc[:, 2:6]
sns.pairplot(data=data, diag_kind="hist")
plt.show()


# 7
df = dataframe
corr_matrix = df.iloc[:, 2:6].corr()

max_corr = abs(corr_matrix.values[np.triu_indices_from(corr_matrix, 1)]).max()
idx = np.where(abs(corr_matrix) == max_corr)
max_corr_features = [corr_matrix.columns[idx[0][0]], corr_matrix.columns[idx[1][0]]]
print('Ознаки з найбільшою кореляцією:', max_corr_features)
print(corr_matrix)

sns.relplot(x="averagespread", y=max_corr_features[0], data=df, hue="productname", palette="Set2", legend=True)
sns.relplot(x="averagespread", y=max_corr_features[1], data=df, hue="productname", palette="Set2", legend=True)
plt.show()


# 8
print("Початок: ", dataframe.isnull().sum())
dataframe.dropna()
dataframe.fillna(0.0)
print("\nПісля: ",dataframe.isnull().sum())


# 9
df_with_10 = df.drop(['productname'], axis=1)
df_without_10 = df_with_10.iloc[np.random.choice(df_with_10.shape[0], round(df_with_10.shape[0] * 0.9))]
print(df_without_10)

numeric_columns = df_with_10.select_dtypes(include=['float64']).columns
df_with_10_numeric = df_with_10[numeric_columns]
df_without_10_numeric = df_without_10[numeric_columns]
print(df_without_10_numeric)

plt.scatter("atlantaretail", "chicagoretail", data=df_without_10_numeric.iloc[:100, :])
plt.xlabel("atlantaretail")
plt.ylabel("chicagoretail")
plt.legend()
plt.show()
