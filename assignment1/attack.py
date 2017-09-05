import numpy as np

def create_dataset(n):
    assert n >= 0
    return np.random.random_integers(0, 1, n)

def sum_query(D, rows):
    return np.sum([D[i] for i in rows])

def sum_query_noisy(D, rows, E):
    pass

def attack(D):
    pass

if __name__ == '__main__':
    D = create_dataset(10)

    print(D)
    print(sum_query(D, [3, 5, 9]))
