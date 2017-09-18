# Машинное обучение.
# Лаб. 1. Метрические алгоритмы классификации

from data import get_data
from naive_bayes import naive_bayes
from nearest_neighbors import nearest_neighbors

import time
 
class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
         
    def __exit__(self, type, value, traceback):
        print("Время выполнения: {:.3f} sec".format(time.time() - self._startTime))

def main():
    print('Сбор данных:')
    d = get_data()
    print('-------------')
    print('Наивный Байесовский классификатор:')
    naive_bayes(*d)
    print('-------------')
    print('Метод ближайших соседей:')

    with Profiler() as p:
        nearest_neighbors(*d, 5)

main()
