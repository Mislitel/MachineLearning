import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Загрузка данных
def load_data(filename):
    print('Загрузка данных из файла...')
    return pd.read_csv(filename, header = None).values

# Разделение датасета
def split_data(data):
    print('Разделение набора данных...')
    attributes = data[:, :-1]
    classes = np.ravel(data[:, -1:].astype(np.int64, copy=False))
    return train_test_split(
        attributes, classes, test_size=0.3, random_state=42)

# Получение данных
def get_data():
    return split_data(load_data('data/spambase.data'))

# Определение точности вычислений
def accuracy(y_result, y_test):
    c = 0
    for i in range(len(y_test)):
        if(y_result[i] == y_test[i]):
            c += 1
    return c / float(len(y_test))
