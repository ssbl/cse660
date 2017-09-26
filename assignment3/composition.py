from collections import defaultdict

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

def point_queries(D, ylist):
    return counting_query(D, ylist)

def attribute_mean_query(D, y):
    y_index = tuple_to_int(y) % 3
    result = [d[y_index] for d in D]

    return sum(result) / len(D)

def attribute_mean_queries(D, ylist):
    return [attribute_mean_query(D, y) for y in ylist]

def tuple_to_int(tup):
    return int(''.join(map(str, tup)), 2)

def threshold_query(D, y):
    y_int = tuple_to_int(y)
    result = [1 for d in D if tuple_to_int(d) <= y_int]

    return sum(result) / len(D)

def threshold_queries(D, ylist):
    return [threshold_query(D, y) for y in ylist]

def lapmech(D, qD, e):
    b = 1.0 / (len(D) * e)
    return qD + np.random.laplace(scale=b)

def round_float(x, n):
    fmt = '{:.' + str(n) +  'f}'
    string = fmt.format(x)
    return float(string)

def rounded_composition(D, query, ylist, budget, digits=3):
    results = composition(D, query, ylist, budget)
    return tuple(round_float(result, digits) for result in results)

def composition(D, query, ylist, budget):
    e = budget / len(ylist)
    query_results = []

    qDs = query(D, ylist)
    for qD in qDs:
        query_results.append(lapmech(D, qD, e))

    return query_results

def privacy_loss(queries=counting_queries):
    n = 200
    iterations = 100
    y, y_ = create_dataset_pair(5) # y_ unused
    # y = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    #      (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    D, D_ = create_dataset_pair(n)
    budget = 4
    digits = 2

    counts_D = defaultdict(int)
    results_D = [rounded_composition(D, queries, y, budget, digits)
                 for _ in range(iterations)]

    print(results_D[:10])

    for result in results_D:
        counts_D[result] += 1
    print(counts_D)

    counts_D_ = defaultdict(int)
    results_D_ = [rounded_composition(D_, queries, y, budget, digits)
                 for _ in range(iterations)]

    for result in results_D_:
        counts_D_[result] += 1
    print(counts_D_)

    ratios = {}
    for result_D in counts_D:
        if counts_D_[result_D]:
            ratios[result_D] = np.log(counts_D[result_D] / counts_D_[result_D])
    print(ratios.values())

    line = plt.axhline(budget, color='r')
    plt.axhline(-budget, color='r')
    plt.legend([line], ['budget = {}'.format(budget)])
    plt.plot(list(ratios.values()))
    plt.show()

if __name__ == '__main__':
    # e = 0.5
    # n = 10
    # iterations = 100
    # y, y_ = create_dataset_pair(5) # y_ unused
    # # y = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    # #      (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    # D, D_ = create_dataset_pair(n)
    # budget = 1.5

    # print(D)
    # print(attribute_mean_query(D, 1))
    privacy_loss(attribute_mean_queries)
