import math
import numpy as np
import pandas as pd

from data import get_data
from data import accuracy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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
def nb_classification(x_test, ms_dict):
    print('- определение наиболее подходящих классов...')
    y_result = []
    for x in x_test:
        y_result.append(get_class_number(x, ms_dict))
    return np.array(y_result)

# Основная функция Наивного Байесовского классификатора
def naive_bayes(x_train, x_test, y_train, y_test):
    # Обучение
    print('Обучение...')
    c_dict = class_dictionary(x_train, y_train)
    p_c = class_apriory(c_dict)
    ms_dict = mu_sigma_by_class(c_dict)
    
    # Тестирование
    print('Тестирование...')
    y_result = nb_classification(x_test, ms_dict)
    print('Точность разработанного ПО: ', accuracy(y_result, y_test))

    # Сравнение со стандартными инструментами
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print('Точность стандартных средств: ', clf.score(x_test, y_test))
    return



def main():
    naive_bayes(*get_data())

main()
