import numpy as np
import matplotlib.pyplot as plt

def create_dataset_pair(n):
    D1 = np.random.randint(2, size=n)
    D2 = np.random.randint(2, size=n)
    D3 = np.random.randint(2, size=n)
    D = list(zip(D1, D2, D3))

    k = np.random.randint(0, n)
    D_ = D[:]

    while D_[k] == D[k]:
        D_[k] = tuple(np.random.randint(2, size=3))

    return D, D_

def counting_query(D, y):
    result = [1 for d in D if d == y]
    return sum(result) / len(D)

def counting_queries(D, ylist):
    return [counting_query(D, y) for y in ylist]

def lapmech(D, qD, e):
    b = 1.0 / (len(D) * e)
    return qD + np.random.laplace(scale=b)

def composition(D, query, e, ylist, budget):
    query_results = []

    qDs = query(D, ylist)
    for qD in qDs:
        if e <= budget:
            query_results.append(lapmech(D, qD, e))
            budget -= e
        else:
            break

    return query_results

if __name__ == '__main__':
    e = 0.5
    n = 10000
    y, y_ = create_dataset_pair(5)
    D, D_ = create_dataset_pair(n)
    budget = 2

    print(composition(D, counting_queries, e, y, budget))
