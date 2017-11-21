import itertools
from pprint import pprint

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
    # [sign col1 col2 col3 compl1 compl2 compl3]
    signs = [0, 1]
    columns = itertools.combinations(range(nbits), 3)
    # result = [[1 if i-1 in c else 0 for i in range(nbits+1)]
    #           for c in itertools.islice(columns, nqueries)]
    complements = itertools.product(range(2), repeat=3)
    # for i in range(len(result)):
    #     negated = result[i][:]
    #     negated[0] = 1
    #     result.append(negated)
    result = itertools.product(columns, complements)
    result = list(itertools.product(signs, result))
    return [[x[0]] + list(itertools.chain(*x[1])) for x in result]

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

# def query_3marginal(row, query):
#     if np.dot(row, query[1:]) == 3:
#         return 0 if query[0] else 1
#     return 1 if query[0] else 0

def query_3marginal(row, query):
    result = 1
    for i in range(3):
        col = query[1 + i]
        complement = query[4 + i]
        result &= int(row[col]) ^ complement
    return result ^ 1 if query[0] else result

def query_3marginal_db(D, query):
    return sum([query_3marginal(row, query) for row in D]) / len(D)

def payoff(D, q, x):
    return query_3marginal(x, q) - query_3marginal_db(D, q)

if __name__ == '__main__':
    # 1. fix payoff DONE
    # 2. manually choose queries
    n = 5000
    nbits = 10
    nqueries = nC3(nbits)
    D = create_dataset(n, nbits)
    # frequencies = get_frequencies(D, nbits)
    Q = create_queries(nqueries, nbits)
    Qdist = np.array([1/len(Q) for _ in range(len(Q))])
    epsilon = 0.5
    beta = 0.05
    delta = np.exp(-10)
    alpha = 2.5
    # steps = (16 * np.log(len(Q))) / alpha**2
    # eta = alpha / 4
    # samples = (48 * np.log(2 * (2**nbits) * steps / beta)) / alpha**2
    steps = 10
    eta = 3.0
    samples = 5
    print('steps =', steps, 'eta =', eta, 'samples =', samples)

    synthetic_db = []
    for t in range(steps):
        print('step {}...'.format(t), end='')
        sampled_queries = np.array(sample_queries(Q, Qdist, samples))
        print('sampled queries')
        pprint(sorted(map(tuple, sampled_queries)))
        # print(Qdist)
        # print('..............................')

        npositive = nnegative = 0
        for query in sampled_queries:
            if query[0]:
                nnegative += 1
            else:
                npositive += 1
        # print('positive =', npositive, 'negative =', nnegative)

        model = pulp.LpProblem('Dual Query', pulp.LpMaximize)
        x = np.array([pulp.LpVariable('x' + str(i), cat='Binary') for i in range(nbits)])
        c = np.array([pulp.LpVariable('c' + str(i), cat='Binary') for i in range(npositive)])
        d = np.array([pulp.LpVariable('d' + str(i), cat='Binary') for i in range(nnegative)])
        # print(model)

        model += sum(c) + sum(d), 'Objective function'

        countp = countn = 0
        for query in sampled_queries:
            vars = []
            for i in range(3):
                col = query[1 + i]
                complement = query[4 + i]
                vars.append(1 - x[col] if complement else x[col])
            if not query[0]:
                # model += np.dot(query[1:], x) - 3 * c[countp] >= 0
                model += sum(vars) - 3 * c[countp] >= 0
                countp += 1
            else:
                # model += np.dot(query[1:], -x) - d[countn] + 3 >= 0
                model += -sum(vars) - d[countn] + 3 >= 0
                countn += 1
        model.solve()
        # Using valueOrDefault to avoid dealing with None values in xt
        xt = np.array([xvar.valueOrDefault() for xvar in x])

        for i in range(len(Q)):
            Qdist[i] = np.exp(-eta * payoff(D, Q[i], xt)) * Qdist[i]
        psum = sum(Qdist)
        # print(psum)
        Qdist /= psum
        # print(sum(Qdist))

        synthetic_db.append(xt)
        # print(model)
        print('done.')

    pprint(synthetic_db)

    # result_D = result_synthetic = []
    result = []
    for query in Q:
        diff = query_3marginal_db(D, query) - query_3marginal_db(synthetic_db, query)
        result.append(diff)
        # result_D.append(query_3marginal_db(D, query))
        # result_synthetic.append(query_3marginal_db(synthetic_db, query))

    # plt.plot(result_D, color='g')
    # plt.plot(result_synthetic, color='r')
    plt.plot(result)
    plt.show()
    # for xvar in x:
    #     print(xvar.varValue)
    # print('=== c ===')
    # for cvar in c:
    #     print(cvar.varValue)
    # print('=== d ===')
    # for dvar in d:
    #     print(dvar.varValue)
