# Машинное обучение.
# Лаб. 1. Метрические алгоритмы классификации

from data import get_data
from naive_bayes import naive_bayes
from nearest_neighbors import nearest_neighbors

def main():
    print('Сбор данных:')
    d = get_data()
    print('-------------')
    print('Наивный Байесовский классификатор:')
    naive_bayes(*d)
    print('-------------')
    print('Метод ближайших соседей:')
    nearest_neighbors(*d, 5)

main()
