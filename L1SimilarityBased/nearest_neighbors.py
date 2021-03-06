import numpy as np

from math import sqrt
from data import accuracy
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# Расчёт расстояния между двумя объектами
def distance(x1, x2):
    return sqrt(sum([(i - j)*(i - j) for i, j in zip(x1, x2)]))

# Расчёт расстояний для каждого объекта
def get_neighbors(x, x_train, y_train, k):
    return sorted(tuple(zip([distance(x, i) for i in x_train], y_train)))[:k]

# Получение списка соседей для каждого объекта
def get_neighbors_list(x_train, y_train, x_test, k):
    print('Получение списка соседей (время выполнения ~ 3 мин)...')
    return [get_neighbors(x, x_train, y_train, k) for x in x_test]

# Получение списка самых частых классов среди соседей
def nn_classification(neighbors):
    print('Классификация...')
    return [Counter(neighbor).most_common()[0][0][1] for neighbor in neighbors]

# Основной алгоритм метода ближайших соседей
def nearest_neighbors(x_train, x_test, y_train, y_test, k):
    # Обучение и тестирование
    neighbors = get_neighbors_list(x_train, y_train, x_test, k)
    y_result = nn_classification(neighbors)
    print('Точность разработанного ПО: {:.3f}'.format(accuracy(y_result, y_test)))

    # Сравнение со стандартными инструментами
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)
    print('Точность стандартных средств: {:.3f}'.format(clf.score(x_test, y_test)))
    return
