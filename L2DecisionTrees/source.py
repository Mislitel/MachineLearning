# Машинное обучение.
# Лаб. 2. Деревья решений

import numpy as np
import pandas as pd

from decimal import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных
def load_data(filename):
    print('Загрузка данных из файла...')
    return pd.read_csv(filename, header = None).values

# Разделение данных
def split_data(data, test_size):
    attributes = data[:, :-1]
    classes = np.ravel(data[:, -1:].astype(np.int64, copy=False))
    return train_test_split(
        attributes, classes, test_size=test_size, random_state=42)

# Классификатор рандомного леса
def random_forest_test(d, c):
    test_size = Decimal(0.4)
    rfc = RandomForestClassifier()
    print('Ramdom Forest Classifier')
    for i in range(c):
        x_train, x_test, y_train, y_test = split_data(d, float(test_size))
        rfc.fit(x_train, y_train)
        result = rfc.score(x_test, y_test)
        print('#', i + 1, ' - тестовая выборка: ', int(test_size*100),
              '%; обучающая выборка: ', int((1 - test_size)*100),
              '%; результат: {:.3f}'.format(result), sep='')
        test_size -= Decimal(0.1)

# Классификатор дерева решений
def decision_tree_test(d, c):
    test_size = Decimal(0.4)
    dtc = DecisionTreeClassifier()
    print('Decision Tree Classifier')
    for i in range(c):
        x_train, x_test, y_train, y_test = split_data(d, float(test_size))
        dtc.fit(x_train, y_train)
        result = dtc.score(x_test, y_test)
        print('#', i + 1, ' - тестовая выборка: ', int(test_size*100),
              '%; обучающая выборка: ', int((1 - test_size)*100),
              '%; результат: {:.3f}'.format(result), sep='')
        test_size -= Decimal(0.1)

def main():
    d = load_data('data/spambase.data')
    c = 4    # Количество экспериментов
    getcontext().prec = 1
    random_forest_test(d, c)
    decision_tree_test(d, c)
    
main()
