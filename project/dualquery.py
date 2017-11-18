import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def create_dataset(n,nbits):
    columns = []
    for i in range(nbits):
        columns.append(np.random.randint(2, size=n))

    D = list(zip(*columns))
    return D

def create_queries(nqueries, nbits):
    columns = itertools.combinations(range(nbits), 3)
    result = [[1 if i-1 in c else 0 for i in range(nbits+1)]
              for c in itertools.islice(columns, nqueries)]
    for i in range(len(result)):
        negated = result[i][:]
        negated[0] = 1
        result.append(negated)
    return result

def get_frequencies(D, nbits):
    result = [0 for _ in range(2**nbits)]
    for x in D:
        result[int(''.join(map(str, x)), 2)] += 1 / len(D)
    return result

def nC3(n):
    return (n * (n-1) * (n-2)) // 6

def sample_queries(Q, Qdist, samples):
    result = []
    queries = list(range(len(Q)))
    for i in range(int(samples)):
        result.append(Q[np.random.choice(queries, p=Qdist)])
    return result

if __name__ == '__main__':
    n = 5000
    nbits = 10
    nqueries = nC3(nbits)
    # D = create_dataset(n, nbits)
    # frequencies = get_frequencies(D, nbits)
    Q = create_queries(nqueries, nbits)
    Qdist = [1/len(Q) for _ in range(len(Q))]
    epsilon = 0.5
    beta = 0.05
    delta = np.exp(-10)
    alpha = 2.5
    steps = (16 * np.log(len(Q))) / alpha**2
    eta = alpha / 4
    samples = (48 * np.log(2 * (2**nbits) * steps / beta)) / alpha**2
    print('steps =', steps, 'eta =', eta, 'samples =', samples)

    # for t in range(steps):
    sampled_queries = sample_queries(Q, Qdist, samples)
    npositive = nnegative = 0
    for query in sampled_queries:
        if query[0]:
            nnegative += 1
        else:
            npositive += 1
    print('positive =', npositive, 'negative =', nnegative)

    matrix = []
    rhs_vector = []
    count_pos, count_neg = 1, 1
    for query in sampled_queries:
        if not query[0]:
            equation = query[1:] + ([0] * (count_pos - 1)) + [-3] \
                       + ([0] * (npositive - count_pos)) + \
                       ([0] * nnegative)
            count_pos += 1
            rhs_vector.append(0)
            matrix.append(equation)
        else:
            equation = [x*-1 for x in query[1:]] + ([0] * npositive) + \
                       ([0] * (count_neg - 1)) + [-1] + \
                       ([0] * (nnegative - count_neg))
            rhs_vector.append(-3)
            count_neg += 1
            matrix.append(equation)

    rhs_vector = -np.array(rhs_vector)
    matrix = -np.array(matrix)
    print(rhs_vector)
    print(sampled_queries[1])
    print(matrix[1])

    c = ([0] * nbits) + ([-1] * int(samples))
    A = matrix
    b = rhs_vector
    bounds_x = [[0, 1] for _ in range(nbits)]
    bounds_d = [[0, 3] for _ in range(nnegative)]
    bounds_c = [[0, 1] for _ in range(npositive)]
    bounds_vector = bounds_x + bounds_c + bounds_d
    res = linprog(c, A_ub=A, b_ub=rhs_vector, bounds=(bounds_vector),
                  options={'disp': True})
    print(res)
