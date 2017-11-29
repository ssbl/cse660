'''
Assignment 11: local sensitivity

References: Salil Vadhan notes section 3.1
'''

import numpy as np
import matplotlib.pyplot as plt

def ls(D, q):
    # Here D is of the form {1..10}^n
    neighbors = []
    for i in range(len(D)):
        for x in range(3):
            if x == D[i]:
                continue
            neighbor = D[:]         # create a copy of D
            neighbor[i] = x
            neighbors.append(neighbor)

    maxdiff = 0
    for neighbor in neighbors:
        maxdiff = max(maxdiff, abs(q(D) - q(neighbor)))

    return maxdiff

def lapmech(D, qD, e, sens):
    b = sens / e
    return qD + np.random.laplace(scale=b)

if __name__ == '__main__':
    # Run queries on random datasets of size 50..1000 in steps of 50.
    # Compute the error for both approaches.
    nlist = list(range(50, 1001, 50))
    queries = [np.median, np.average]
    epsilon = 1.0

    error_gs = []
    error_ls = []

    for n in nlist:
        D = np.random.random_integers(1, 10, n).tolist()
        for query in queries:
            query_desc = 'median' if query is np.median else 'average'
            qD = query(D)
            lsval = ls(D, query)
            gs = 10 if query is np.median else 10 / n
            print('Running LapMech for n = {}, query = {}, LS = {}'.format(
                n, query_desc, lsval))
            error_gs.append(abs(qD - lapmech(D, qD, epsilon, gs)))
            error_ls.append(abs(qD - lapmech(D, qD, epsilon, ls(D, query))))

    # Error when using LS turns out to be very low.
    plt.plot(error_gs, 'ro', label='global')
    plt.plot(error_ls, 'go', label='local')
    plt.legend()
    plt.show()
