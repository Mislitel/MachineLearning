import math
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

# Ирисы
def get_iris():
    print('Получение данных об ирисах...')
    iris = load_iris()
    x = iris.data
    y = iris.target
    return train_test_split(
        x, y, test_size=0.33, random_state=42)

# Числа
def get_digits():
    print('Получение данных о цифрах...')
    iris = load_digits()
    x = iris.data
    y = iris.target
    return train_test_split(
        x, y, test_size=0.33, random_state=42)

# Загрузка данных
def load_data(filename):
    print('Загрузка данных из файла...')
    return pd.read_csv(filename, header = None).values
    #return pd.read_csv(filename, sep=',').values

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



# Определение математического ожидания и дисперсии каждого атрибута
def mu_sigma(X):
    m = [sum(x) / len(x) for x in X]
    d = [sum(map(lambda y: y**2, x)) / len(x)
         - pow(sum(x) / len(x), 2) for x in X]
    return m, d

# Функция правдоподобия
def liklihood(x, m, d):
    if d == 0:
        d = 0.000001
    return 1./(d * math.sqrt(2 * math.pi)) * math.exp(-pow(x - m, 2)/(2 * d**2))

# Получение мат. ожидания и дисперсии для каждого класса
def mu_sigma_by_class(c_dict):
    print('- определение мат. ожидания и дисперсии для классов...')
    ms_dict = {}
    for class_name, attributes in c_dict.items():
        ms_dict[class_name] = mu_sigma(attributes)
    return ms_dict

# Формирование словаря классов
def class_dictionary(attributes, classes):
    print('- формирование словаря классов...')
    temp = {}
    c_dict = {}
    for i in range(len(attributes)):
        temp.setdefault(classes[i], []).append(attributes[i])
    for key, value in temp.items():
        c_dict[key] = np.asarray(value).transpose()
    return c_dict

# Получение априорных вероятностей для всех классов
def class_apriory(c_dict):
    print('- получение априорных вероятностей для классов...')
    train_size = float(sum([len(x[0]) for x in list(c_dict.values())]))
    return [len(x[0]) / train_size for x in list(c_dict.values())]

# Определение наиболее подходящего класса для вектора атрибутов х
def get_class_number(x, ms_dict):
    result = 0
    pre_p = 0
    for key, summary in ms_dict.items():
        m, d = summary
        p = 1
        for i in range(len(x)):
            p *= liklihood(x[i], m[i], d[i])
        if p > pre_p:
            result = key
            pre_p = p
    return result

# Определение вероятности принадлежности объекта к каждому классу
def classification(x_test, ms_dict):
    print('- определение наиболее подходящих классов...')
    y_result = []
    for x in x_test:
        y_result.append(get_class_number(x, ms_dict))
    return np.array(y_result)

# Определение точности вычислений
def accuracy(y_result, y_test):
    c = 0
    for i in range(len(y_test)):
        if(y_result[i] == y_test[i]):
            c += 1
    return c / float(len(y_test))

# Основная функция Наивного Байесовского классификатора
def naive_bayes(x_train, x_test, y_train, y_test):
    # Обучение
    print('Обучение...')
    c_dict = class_dictionary(x_train, y_train)
    p_c = class_apriory(c_dict)
    ms_dict = mu_sigma_by_class(c_dict)
    
    # Тестирование
    print('Тестирование...')
    y_result = classification(x_test, ms_dict)
    print('Точность разработанного ПО: ', accuracy(y_result, y_test))

    # Сравнение со стандартными инструментами
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print('Точность стандартных средств: ', clf.score(x_test, y_test))
    return



def main():
    #naive_bayes(*get_iris())
    #naive_bayes(*get_digits())
    naive_bayes(*get_data())

main()
