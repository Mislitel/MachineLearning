import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Ирисы
def l1_iris():
    print('Получение данных об ирисах...')
    iris = load_iris()
    x = iris.data
    y = iris.target
    return train_test_split(
        x, y, test_size=0.33, random_state=42)

# Определение математического ожидания и дисперсии каждого атрибута
def mu_sigma(X):
    m = [sum(x) / len(x) for x in X]
    d = [sum(map(lambda y: y**2, x)) / len(x)
         - pow(sum(x) / len(x), 2) for x in X]
    return m, d

# Функция правдоподобия
def liklihood(x, m, d):
    return 1./(d * math.sqrt(2 * math.pi)) * math.exp(-pow(x - m, 2)/(2 * d**2))

# Получение мат. ожидания и дисперсии для каждого класса
def mu_sigma_by_class(c_dict):
    ms_dict = {}
    for class_name, attributes in c_dict.items():
        ms_dict[class_name] = mu_sigma(attributes)
    return ms_dict

# Формирование словаря классов
def l1_class_dictionary(attributes, classes):
    print('- Формирование словаря классов...')
    temp = {}
    c_dict = {}
    for i in range(len(attributes)):
        temp.setdefault(classes[i], []).append(attributes[i])
    for key, value in temp.items():
        c_dict[key] = np.asarray(value).transpose()
    return c_dict

# Получение априорных вероятностей для всех классов
def l1_class_apriory(c_dict):
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
    y_result = []
    for x in x_test:
        y_result.append(get_class_number(x, ms_dict))
    return np.array(y_result)

# Основная функция Наивного Байесовского классификатора
def l1_naive_bayes(x_train, x_test, y_train, y_test):
    # Обучение
    c_dict = l1_class_dictionary(x_train, y_train) # Разбили на классы
    p_c = l1_class_apriory(c_dict)  # Определили априорные вероятности классов
    ms_dict = mu_sigma_by_class(c_dict) # Определили мат. ожидание и дисперсию для каждого класса
    
    # Тестирование
    y_result = classification(x_test, ms_dict)  # Получение результатов классификации
    # Добавить вычисление точности
    return



def main():
    l1_naive_bayes(*l1_iris())

main()
