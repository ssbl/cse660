'''
Assignment 11: local sensitivity

References: Salil Vadhan notes section 3.1
'''

import numpy as np
import matplotlib.pyplot as plt

def ls(D, q, lowerbound, upperbound):
    n = len(D)
    # Here D is of the form {lower..upper}^n
    neighbors = []
    for i in range(n):
        for x in range(lowerbound, upperbound + 1):
            if x != D[i]:
                neighbor = D.copy()
                neighbor[i] = x # change a single row
                neighbors.append(neighbor)

    maxdiff = 0
    for neighbor in neighbors:
        diff = abs(q(D) - q(neighbor))
        maxdiff = max(maxdiff, abs(q(D) - q(neighbor)))

    return maxdiff

def lapmech(D, qD, e, sens):
    b = sens / e
    return qD + np.random.laplace(scale=b)

if __name__ == '__main__':
    # Run queries on random datasets of size 50..500 in steps of 50.
    # Compute the error for both approaches.
    lowerbound, upperbound = 1, 10
    nlist = list(range(50, 501, 50))
    queries = [np.median, np.average]
    epsilon = 1.0

    error_gs = []
    error_ls = []

    xs = []
    elemrange = upperbound - lowerbound
    for n in nlist:
        D = np.random.random_integers(lowerbound, upperbound, n).tolist()
        for query in queries:
            xs.append(n)
            if query is np.median:
                query_desc = 'median'
            else:
                query_desc = 'average'
            qD = query(D)
            lsval = ls(D, query, lowerbound, upperbound)
            gs = elemrange if query is np.median else elemrange / n
            print('Running LapMech for n = {}, query = {}, LS = {}'.format(
                n, query_desc, lsval))
            error_gs.append(abs(qD - lapmech(D, qD, epsilon, gs)))
            error_ls.append(abs(qD - lapmech(D, qD, epsilon, lsval)))

    plt.plot(xs, error_gs, 'ro', label='global')
    plt.plot(xs, error_ls, 'go', label='local')
    plt.xlabel('number of rows')
    plt.ylabel('error')
    plt.legend()
    plt.show()
