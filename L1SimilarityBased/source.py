# Машинное обучение.
# Лаб. 1. Метрические алгоритмы классификации

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB   # Используется в случае дискретных признаков (наш случай)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris

# Загрузка данных
def ml_load(filename):
    print('Загрузка данных из файла...')
    return pd.read_csv(filename, header = None).values

# Разделение датасета
def ml_split(data):
    '''
    print('Разделение набора данных...')
    attributes = data[:, :-1]
    classes = np.ravel(data[:, -1:])
    return train_test_split(
        attributes, classes, test_size=0.33, random_state=0)
    '''
    iris = load_iris()
    x = iris.data
    y = iris.target
    return train_test_split(
        x, y, test_size=0.33, random_state=42)
        
# Прогон стандартной библиотеки для Наивного Байеса :)
def nb_standard(x_train, y_train, x_test, y_test):
    print('Работа стандартной библиотеки sklearn (Naive Bayes)...')
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))
    print(clf.score(x_test, y_test))

# Прогон стандартной библиотеки для Ближайших Соседей
def nn_standard(x_train, y_train, x_test, y_test):
    print('Работа стандартной библиотеки sklearn (Nearst Neighbors)...')
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))



def main():
    data = ml_load("data/poker-hand-testing.data")
    x_train, x_test, y_train, y_test = ml_split(data)
    
    nb_standard(x_train, y_train, x_test, y_test)
    #nn_standard(x_train, y_train, x_test, y_test)

main()
'''
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
clf = GaussianNB()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
'''
