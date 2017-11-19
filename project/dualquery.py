import itertools

import pulp
import numpy as np
import matplotlib.pyplot as plt


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
    # steps = (16 * np.log(len(Q))) / alpha**2
    # eta = alpha / 4
    # samples = (48 * np.log(2 * (2**nbits) * steps / beta)) / alpha**2
    steps = 14
    eta = 3.0
    samples = 20
    print('steps =', steps, 'eta =', eta, 'samples =', samples)

    # for t in range(steps):
    sampled_queries = np.array(sample_queries(Q, Qdist, samples))

    npositive = nnegative = 0
    for query in sampled_queries:
        if query[0]:
            nnegative += 1
        else:
            npositive += 1
    print('positive =', npositive, 'negative =', nnegative)

    model = pulp.LpProblem('Dual Query', pulp.LpMaximize)
    x = np.array([pulp.LpVariable('x' + str(i), cat='Binary') for i in range(nbits)])
    c = np.array([pulp.LpVariable('c' + str(i), cat='Binary') for i in range(npositive)])
    d = np.array([pulp.LpVariable('d' + str(i), cat='Binary') for i in range(nnegative)])

    model += sum(c) + sum(d), 'Objective function'

    countp = countn = 0
    for query in sampled_queries:
        if not query[0]:
            model += np.dot(query[1:], x) - 3 * c[countp] >= 0
            countp += 1
        else:
            model += np.dot(query[1:], -x) - d[countn] >= -3
            countn += 1
    model.solve()

    for xvar in x:
        print(xvar.varValue)
    print('=== c ===')
    for cvar in c:
        print(cvar.varValue)
    print('=== d ===')
    for dvar in d:
        print(dvar.varValue)
