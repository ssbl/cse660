import numpy as np
import random
import itertools

def create_dataset(n):
    assert n >= 0
    return np.random.random_integers(0, 1, n).tolist()

def sum_query(D, rows):
    return sum([D[i] for i in rows])

def sum_query_noisy(D, rows, E):
    return sum_query(D, rows) + random.uniform(-E, E)

def subset(my_list):
    result = []
    for i in range(1,len(my_list)+1):
        result += list(itertools.combinations(my_list,i))
    return result

def binary_set(n):
    return list(itertools.product([0,1], repeat=n))

# https://stackoverflow.com/a/31007358
def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def attack(D,E):
    n = len(D)
    noisy_query_result = {}
    for i in subset(range(0,n)):
        nq = sum_query_noisy(D,i,E)
        noisy_query_result[i] = nq

    D_dash = binary_set(n)
    for d in D_dash:
        for i in noisy_query_result.keys():
            sq = sum_query(d, i)
            nqr = noisy_query_result[i]
            if abs(sq-nqr) > E:
                break
        else:
            return d

if __name__ == '__main__':
    n = 8
    E = 1
    for i in range(100):
        D = create_dataset(n)
        D_dash =  attack(D,E)
        hamDist = hamming2(D_dash,D)
        print(D,D_dash,hamDist)
