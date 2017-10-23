import numpy as np
import matplotlib.pyplot as plt
import itertools

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

def tuple2int(tup):
    return int(''.join(map(str, tup)), 2)

def counting_query(D, y):
    return D[tuple2int(y)]

def create_universe(nbits):
    a = np.arange(2**nbits)
    return [tuple(map(int,bin(d)[2:].rjust(nbits,'0'))) for d in a]

def score(D, D_, q):
    return abs(counting_query(D, q) - counting_query(D_, q))

def generate_probabilities(D, D_, u, R, epsilon):
    delta_u = 1 / len(D)
    scores = [u(D, D_, r) for r in R]
    results = [np.exp((epsilon * u(D, D_, r)) / (2.0 * delta_u)) for r in R]
    # print(results)
    total = sum(results)
    return [x / total for x in results]

def get_frequencies(D, nbits):
    result = [0 for _ in range(2**nbits)]
    for x in D:
        result[int(''.join(map(str, x)), 2)] += 1 / len(D)
    return result

if __name__ == '__main__':
    epsilon = 0.5
    nbits = 8
    nuniverse = 2**nbits
    n = 1000
    nqueries = 205               # m
    alpha = 0.1
    T = 1 + np.log(nuniverse) / (alpha**2)
    print('T = ' + str(T))
    D = create_dataset_pair(n,nbits)
    Q = create_queries(nqueries, nbits);
    print(Q)

    distribution0 = [1 / 2**nbits for _ in range(2**nbits)]
    frequencies = get_frequencies(D, nbits) # this is D

    epsilon_exp = (n * epsilon) / (2 * T)
    scale = T / (n * epsilon)
    print(epsilon_exp)
    print('scale = {}, random value = {}'.format(scale,
                                                 np.random.laplace(scale=scale)))
    frequencies_ = distribution0 # this is D_{i-1}
    frequency_list = []

    for _ in range(int(T)):
        frequency_list.append(frequencies_)
        p = generate_probabilities(frequencies, frequencies_,
                                   score, Q, epsilon_exp)
        q_cap = Q[np.random.choice(range(len(Q)), p=p)]
        m = counting_query(frequencies, q_cap) + np.random.laplace(scale=scale)
        frequencies_i = frequencies_ # this is D_i
        frequencies_i[tuple2int(q_cap)] = \
          frequencies_[tuple2int(q_cap)] * np.exp(0.5 * (m - counting_query(frequencies_, q_cap)))
        frequencies_ = frequencies_i
        total = sum(frequencies)
        frequencies_ = [x / total for x in frequencies_]

    result_frequency = [sum(x) / len(x) for x in zip(*frequency_list)]

    S = create_queries(nuniverse, nbits)
    print(S)
    query_result_original = [counting_query(frequencies, q) for q in S]
    query_result_mwem = [counting_query(result_frequency, q) for q in S]

    comparison = plt.figure(0)
    orig_line = plt.plot(query_result_original, label='original')
    mwem_line = plt.plot(query_result_mwem, label='MWEM')
    comparison.legend()
    comparison.show()

    error = plt.figure(1)
    difference = [x - y for (x, y) in zip(query_result_original,
                                          query_result_mwem)]
    diff_line = plt.plot(difference, 'go', label='error')
    alpha_line = plt.axhline(alpha, label='alpha', color='red')
    plt.axhline(-alpha, color='red')
    error.legend()
    error.show()
    input()
