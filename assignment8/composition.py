import numpy as np
import matplotlib.pyplot as plt

from mwem import mwem_query_result

def create_dataset_pair(n,nbits):
    columns = []
    for i in range(nbits):
        columns.append(np.random.randint(2, size=n))

    D = list(zip(*columns))
    return D

def create_queries(n,nbits):
    a = np.arange(2**nbits)
    np.random.shuffle(a)
    return [tuple(map(int,bin(d)[2:].rjust(nbits,'0'))) for d in a[:n]]

def counting_query(D, y):
    result = [1 for d in D if d == y]
    return sum(result) / len(D)

def counting_queries(D, ylist):
    return [counting_query(D, y) for y in ylist]

def lapmech(D, qD, e):
    b = 1.0 / (len(D) * e)
    return qD + np.random.normal(scale=b)

def composition(D, query, ylist, budget):
    e = budget / len(ylist)
    print('composition:', e)
    query_results = []

    qDs = query(D, ylist)
    for qD in qDs:
        query_results.append(lapmech(D, qD, e))

    return query_results

def advanced(D, query, ylist, epsilon_, delta_):
    k = len(ylist)
    e = epsilon_ / (2 * np.sqrt(2 * k * np.log(1 / delta_)))
    print('advanced:', e)
    query_results = []

    qDs = query(D, ylist)
    for qD in qDs:
        query_results.append(lapmech(D, qD, e))

    return query_results

if __name__ == '__main__':
    n = 1000
    nbits = 8
    budget = 1
    nqueries = 200
    delta_ = np.exp(-5)

    D = create_dataset_pair(n, nbits)

    error_standard = []
    error_advanced = []
    error_mwem = []
    for nq in range(50, 250, 5):
        Q = create_queries(nq, nbits)
        qDs = counting_queries(D, Q)
        qDs_composition = composition(D, counting_queries, Q, budget)
        qDs_advanced = advanced(D, counting_queries, Q, budget, delta_)
        qDs_mwem = mwem_query_result(D, Q, budget)

        diffs_comp_max = max(np.abs(np.array(qDs) - np.array(qDs_composition)))
        error_standard.append(diffs_comp_max)
        error_advanced_max = max(np.abs(np.array(qDs) - np.array(qDs_advanced)))
        error_advanced.append(error_advanced_max)
        error_mwem_max = max(np.abs(np.array(qDs_mwem) - np.array(qDs)))
        error_mwem.append(error_mwem_max)

    # plt.plot(qDs_mwem, label='mwem')
    # plt.plot(qDs, label='none')
    # plt.plot(qDs_composition, label='standard')
    # plt.plot(qDs_advanced, label='advanced')
    plt.plot(error_standard, label='standard')
    plt.plot(error_advanced, label='advanced')
    plt.plot(error_mwem, label='MWEM')
    plt.legend()
    plt.show()
