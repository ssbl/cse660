import itertools
import json
from pathlib import Path
from pickle import dump, load
from pprint import pprint, pformat
from sys import argv
from time import time, strftime

import pulp
import numpy as np
import matplotlib.pyplot as plt


cache = {}

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

def query_3marginal_db_cached(query):
    return cache[query]

def payoff(D, q, x):
    return query_3marginal(x, q) - query_3marginal_db_cached(q)

def print_result(x, c, d):
    for xvar in x:
        print(xvar.varValue)
    print('=== c ===')
    for cvar in c:
        print(cvar.varValue)
    print('=== d ===')
    for dvar in d:
        print(dvar.varValue)

def get_queries(nqueries, nbits):
    result = []

    signs = [0, 1]
    columns = set()
    while len(columns) < nqueries:
        triple = set()
        while len(triple) < 3:
            triple.add(np.random.randint(nbits))
        triple = sorted(triple)
        columns.add(tuple(triple))
    complements = itertools.product(range(2), repeat=3)
    result = itertools.product(columns, complements)
    result = list(itertools.product(signs, result))
    return [tuple([x[0]] + list(itertools.chain(*x[1]))) for x in result]

def run_experiment(eta, steps, samples, D, Q):
    n = len(D)
    Qdist = np.array([1/len(Q) for _ in range(len(Q))])
    print('steps =', steps, 'eta =', eta, 'samples =', samples)

    synthetic_db = []
    start = time()
    for t in range(steps):
        print('step {:3d}/{}'.format(t + 1, steps), end='\r', flush=True)
        sampled_queries = np.array(sample_queries(Q, Qdist, samples))

        npositive = nnegative = 0
        for query in sampled_queries:
            if query[0]:
                nnegative += 1
            else:
                npositive += 1

        model = pulp.LpProblem('Dual Query', pulp.LpMaximize)
        x = np.array([pulp.LpVariable('x' + str(i), cat='Binary') for i in range(nbits)])
        c = np.array([pulp.LpVariable('c' + str(i), cat='Binary') for i in range(npositive)])
        d = np.array([pulp.LpVariable('d' + str(i), cat='Binary') for i in range(nnegative)])

        model += sum(c) + sum(d), 'Objective function'

        countp = countn = 0
        for query in sampled_queries:
            vars = []
            for i in range(3):
                col = query[1 + i]
                complement = query[4 + i]
                vars.append(1 - x[col] if complement else x[col])
            if not query[0]:
                model += sum(vars) - 3 * c[countp] >= 0
                countp += 1
            else:
                model += -sum(vars) - d[countn] + 3 >= 0
                countn += 1
        model.solve()

        # Using valueOrDefault to avoid dealing with None values in xt
        xt = np.array([xvar.valueOrDefault() for xvar in x])

        for i in range(len(Q)):
            Qdist[i] = np.exp(-eta * payoff(D, Q[i], xt)) * Qdist[i]
        psum = sum(Qdist)

        Qdist /= psum

        synthetic_db.append(xt)

    print()
    runtime = time() - start

    result = []
    max_error = avg_error = 0
    for query in Q:
        diff = query_3marginal_db_cached(query) \
               - query_3marginal_db(synthetic_db, query)
        max_error = max(max_error, abs(diff))
        avg_error += abs(diff)
        result.append(diff)

    avg_error /= len(result)
    return {
        'average': avg_error,
        'max': max_error,
        'runtime': runtime
    }

def average_nexperiments(n, start_time, **args):
    avg_error = 0
    max_error = 0
    runtime = 0

    for i in range(1, n + 1):
        print('run {}'.format(i))
        result = run_experiment(**args)
        avg_error += result['average']
        max_error += result['max']
        runtime += result['runtime']

    max_error /= n
    avg_error /= n
    runtime /= n

    with open('log_{}.json'.format(start_time), 'a') as log:
        dump = {
            'steps': args['steps'],
            'eta': args['eta'],
            'samples': args['samples'],
            'max_error': max_error,
            'avg_error': avg_error,
            'runtime': runtime,
            'n': len(args['D'])
        }

        log.write(json.dumps(dump, sort_keys=True) + '\n')

def cache_results(D, Q):
    c = cache
    i = 1
    n = len(Q)
    for query in Q:
        c[query] = query_3marginal_db(D, query)
        print('query {}/{}'.format(i, n), end='\r', flush=True)
        i += 1
    print()

def calculate_nsamples(eps_min, eps_max, eps_steps, eta, steps, nrows):
    eps_increment = (eps_max - eps_min) / (eps_steps - 1)

    result = []

    for i in range(eps_steps):
        current_eps = eps_min + i * eps_increment
        s = (current_eps * nrows) / (eta * steps * (steps - 1))
        result.append(s)

    return result

def calculate_nsteps(eps_min, eps_max, eps_steps, eta, samples, nrows):
    eps_increment = (eps_max - eps_min) / (eps_steps - 1)

    result = set()

    for i in range(eps_steps):
        current_eps = eps_min + i * eps_increment
        x = (current_eps * nrows) / (eta * samples)
        T1, T2 = np.roots([1, -1, -x]).real
        if T1 >= 1:
            result.add(int(T1))
        if T2 >= 1:
            result.add(int(T2))

    return list(result)

if __name__ == '__main__':
    argc = len(argv)
    if argc != 2:
        print('usage: dualquery <pickle-file>')
        exit(1)

    pickle_file = argv[1]
    D = load(open(pickle_file, 'rb'))
    n = len(D)
    nbits = len(D[0])
    nqueries = 1000
    print('n = {}, nbits = {}, nqueries = {}'.format(n, nbits, nqueries))

    cachefile = 'qcache_r-{}_c-{}_q-{}.p'.format(n, nbits, nqueries)
    if Path(cachefile).exists():
        print('Found cache file: {}'.format(cachefile))
        start = time()
        cache = load(open(cachefile, 'rb'))
        Q = list(cache.keys())
        print('Read cache file in {}s'.format(time() - start))
    else:
        print('Generating queries...')
        start = time()
        Q = get_queries(nqueries, nbits)
        print('Created queries in {}s'.format(time() - start))
        print('Creating cache...')
        start = time()
        cache_results(D, Q)
        print('Generated cache in {}s'.format(time() - start))
        with open(cachefile, 'wb') as cf:
            dump(cache, cf)

    t = strftime('%m-%d-%H-%M-%S')
    eta = 0.1
    # steps = 200
    # samples_list = calculate_nsamples(0.1, 5.0, 15, eta, steps, n)
    samples = 50
    steps_list = calculate_nsteps(0.1, 5.0, 15, eta, samples, n)
    for steps in steps_list:
        average_nexperiments(3, t, eta=eta, steps=steps,
                             samples=samples, D=D, Q=Q)

    '''
    eta = 2.7                   # fixed for now
    steps = 12
    steps_max = 80
    steps_increment = 4
    samples = 30
    samples_max = 75
    samples_increment = 13
    # steps = 10
    # steps_max = 22
    # samples = 5
    # samples_max = 20

    # run_experiment(eta=eta, steps=steps, samples=samples, D=D, Q=Q, start_time=t)
    for T in range(steps, steps_max + 1, steps_increment):
        for s in range(samples, samples + 1, samples_increment):
            # run_experiment(eta=eta, steps=T, samples=s, D=D, Q=Q, start_time=t)
            average_nexperiments(3, t, eta=eta, steps=T, samples=s, D=D, Q=Q)
    '''
