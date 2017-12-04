""" This is the main code for Dual Query Algorithm"""
import itertools
import json
from pathlib import Path
from pickle import dump, load
from sys import argv
from time import time, strftime

import pulp
import numpy as np


training_qcache = {}
test_qcache = {}


def create_dataset(n, nbits):
    """ Create Synthetic Dataset

    Args:
        n: Number of rows in dataset
        nbits: Number of columns in dataset

    Returns:
        The dataset as a list of lists. 
    """
    D = []
    for _ in range(n):
        # generate 1 row
        D.append(np.random.randint(2, size=nbits))

    return D

def nC3(n):
    """ Computes n choose 3 """    
    return (n * (n-1) * (n-2)) // 6

def sample_queries(Q, Qdist, samples):
    """ Sample the queries from Query set according to the Query Distribution

    Args:
        Q: Query set
        Qdist: Query Distribution
        samples: Number of queries to be sampled

    Returns:
        List of sampled queries
    """    
    result = []
    # since np.random.choice can only sample from 1D array, we use indices
    queries = list(range(len(Q)))
    for _ in range(int(samples)):
        # pick a random index based on Qdist and take the query at that index
        result.append(Q[np.random.choice(queries, p=Qdist)])
    return result

def query_3marginal(row, query):
    """ Run a 3 marginal query on a row

    Args:
        row: row from dataset
        query: query to run

    Returns:
        1 if row satisfies query, 0 otherwise
    """    
    result = 1
    # Query Format: {sign, column1, column2, column3, complement1, complement2, complement3}
    # run for each column
    for i in range(3):
        # get column number
        col = query[1 + i]
        # get corresponding complement bit (1 if complement 0 if not)
        complement = query[4 + i]
        # the result is 1 only if the bit at column and complement bit differ
        # this can be done by taking XOR
        result &= int(row[col]) ^ complement
    # flip the result if sign bit is set otherwise don't
    return result ^ 1 if query[0] else result

def query_3marginal_db(D, query):
    """ Run a 3 marginal query on Dataset

    Args:
        D: Dataset
        query: query to run

    Returns:
        Normalized result between 0 and 1
    """    
    # run query on each row and normalize the result
    return sum([query_3marginal(row, query) for row in D]) / len(D)

def query_3marginal_db_cached(query, cache):
    """ Run a 3 marginal query on Dataset using cache

    Args:
        query: query to run
        cache: a cached dictionary mapping query to corresponding result on dataset

    Returns:
        Normalized result between 0 and 1
    """    
    return cache[query]

def payoff(D, q, x):
    """ Calculate payoff using equation q(x) - q(D)

    Args:
        D: Dataset
        q: query to run
        x: a row from dataset

    Returns:
        payoff between -1 and 1
    """    
    return query_3marginal(x, q) - query_3marginal_db_cached(q, training_qcache)

def get_queries(nqueries, nbits):
    """ Generate random 3-Marginal queries
    Query Format: {sign, column1, column2, column3, complement1, complement2, complement3}

    Args:
        nqueries: number of queries to be generated
        nbits: number of columns in dataset

    Returns:
        Two lists of queries, first for training(70%) and second for testing(30%)
    """    
    result = []

    signs = [0, 1]
    columns = set()
    # generate unique triplet of unique columns
    while len(columns) < nqueries:
        triple = set()
        while len(triple) < 3:
            triple.add(np.random.randint(nbits))
        triple = sorted(triple)
        columns.add(tuple(triple))

    # split column set into training(70%) and test(30%)
    columns = list(columns)
    split_index = int(0.7 * nqueries)
    training, test = columns[:split_index], columns[split_index:]

    
    results = [training, test]
    for i in range(len(results)):
        datasplit = results[i]
        # generate all complements [0 0 0] to [1 1 1]
        complements = itertools.product(range(2), repeat=3)
        # take cross product of datasplit with complement to concatenate
        # each row of complement to each triplet in datasplit
        result = itertools.product(datasplit, complements)
        # do similar procedure to concatenate sign with result,
        result = list(itertools.product(signs, result))
        # currently each element in result is not flattened but is a list of tuples
        # Now flatten each element in the result so that it conforms to query format
        results[i] = [tuple([x[0]] + list(itertools.chain(*x[1]))) for x in result]

    return results[0], results[1]

def run_experiment(eta, steps, samples, D, Q, Qtest):
    """ Execute a single run of the algorithm for given parameters

    Args:
        eta: learning rate eta
        steps: number of iterations / number of rows generated in output dataset
        samples: number of queries to be sampled from query set
        D: Original Dataset
        Q: Query set
        Qtest: Set of queries to test accuracy

    Returns:
        A dictionary of average error, max error and runtime
    """    

    n = len(D)
    # initialize Query distribution as uniform distribution
    Qdist = np.array([1/len(Q) for _ in range(len(Q))])
    print('steps =', steps, 'eta =', eta, 'samples =', samples)

    synthetic_db = []
    start = time()
    for t in range(steps):
        print('step {:3d}/{}'.format(t + 1, steps), end='\r', flush=True)
        sampled_queries = np.array(sample_queries(Q, Qdist, samples))

        # count number of positive and negative queries
        npositive = nnegative = 0
        for query in sampled_queries:
            if query[0]:
                nnegative += 1
            else:
                npositive += 1

        # create model and variables for the model
        model = pulp.LpProblem('Dual Query', pulp.LpMaximize)
        x = np.array([pulp.LpVariable('x' + str(i), cat='Binary') for i in range(nbits)])
        c = np.array([pulp.LpVariable('c' + str(i), cat='Binary') for i in range(npositive)])
        d = np.array([pulp.LpVariable('d' + str(i), cat='Binary') for i in range(nnegative)])

        model += sum(c) + sum(d), 'Objective function'

        # go through each sampled query and add a constraint to the model
        # depending on the type of query


        countp = countn = 0  #used for tracking index of c and d variables
        for query in sampled_queries:
            vars = []
            for i in range(3):
                col = query[1 + i]
                complement = query[4 + i]
                vars.append(1 - x[col] if complement else x[col])
            if not query[0]:    # query is positive
                model += sum(vars) - 3 * c[countp] >= 0
                countp += 1
            else:
                model += -sum(vars) - d[countn] + 3 >= 0
                countn += 1
        # run the solver
        model.solve()

        # Using valueOrDefault, free variable with value None are set to 0
        xt = np.array([xvar.valueOrDefault() for xvar in x])

        # update the query distribution and normalize
        for i in range(len(Q)):
            Qdist[i] = np.exp(-eta * payoff(D, Q[i], xt)) * Qdist[i]
        psum = sum(Qdist)
        Qdist /= psum

        # add this row to synthetic dataset
        synthetic_db.append(xt)

    print()
    runtime = time() - start

    # calculate maximum error and average error
    result = []
    max_error = avg_error = 0
    for query in Qtest:
        diff = query_3marginal_db_cached(query, test_qcache) \
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

def average_nexperiments(n, start_time, **kwargs):
    """ Run the algorithm for multiple runs and log the average of results

    Args:
        n: number of runs
        start_time: start_time used for logging
        **kwargs: keyword arguments for run_experiment
    """    
    avg_error = 0
    max_error = 0
    runtime = 0

    # run experiment for n number of runs
    for i in range(1, n + 1):
        print('run {}'.format(i))
        result = run_experiment(**kwargs)
        avg_error += result['average']
        max_error += result['max']
        runtime += result['runtime']

    max_error /= n
    avg_error /= n
    runtime /= n

    # write to log file
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

def cache_results(D, Q, Qtest):
    """ Generating cache for Query set and Test Query set using Dataset
    saves results to global variables training_qcache and test_qcache

    Args:
        D: Dataset
        Q: Query set
        Qtest: Test Query set
    """    
    # avoid global lookups for performance
    test_cache = test_qcache
    training_cache = training_qcache

    i = 1
    n = len(Q) + len(Qtest)

    for query in Q:
        training_cache[query] = query_3marginal_db(D, query)
        print('query {}/{}'.format(i, n), end='\r', flush=True)
        i += 1
    for query in Qtest:
        test_cache[query] = query_3marginal_db(D, query)
        print('query {}/{}'.format(i, n), end='\r', flush=True)
        i += 1
    print()

def calculate_nsteps(eps_min, eps_max, step_count, eta, samples, nrows):
    """ Calculate `step_count` number of steps for a given range of epsilon
    using given parameters

    Args:
        eps_min: lower bound of epsilon
        eps_max: upper bound of epsilon
        step_count: number to steps to calculate
        eta: learning rate eta
        samples: number of samples used in algorithm
        nrows: number of rows in original dataset

    Returns:
        a list of generated step values
    """    
    eps_increment = (eps_max - eps_min) / (step_count - 1)

    result = set()

    for i in range(step_count):
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
        print('usage: dualquery <pickle-file>\npickle-file contains original'
            ' binary dataset as list of lists')
        exit(1)

    pickle_file = argv[1]
    D = load(open(pickle_file, 'rb'))
    n = len(D)
    nbits = len(D[0])
    nqueries = 10000
    print('n = {}, nbits = {}, nqueries = {}'.format(n, nbits, nqueries))

    cachefile = 'qcache_r-{}_c-{}_q-{}.p'.format(n, nbits, nqueries)
    # if cache file exists load it, otherwise create cache file and save it
    if Path(cachefile).exists():
        print('Found cache file: {}'.format(cachefile))
        start = time()
        training_qcache, test_qcache = load(open(cachefile, 'rb'))
        Q = list(training_qcache.keys())
        Qtest = list(test_qcache.keys())
        print('Read cache file in {}s'.format(time() - start))
    else:
        print('Generating queries...')
        start = time()
        Q, Qtest = get_queries(nqueries, nbits)
        print('Created queries in {}s'.format(time() - start))
        print('Creating cache...')
        start = time()
        cache_results(D, Q, Qtest)
        print('Generated cache in {}s'.format(time() - start))
        with open(cachefile, 'wb') as cf:
            dump([training_qcache, test_qcache], cf)


    t = strftime('%m-%d-%H-%M-%S')
    eta = 0.1
    samples = 50
    # get list of steps variable
    steps_list = calculate_nsteps(eps_min=0.1, eps_max=5.0, step_count=15,
                                eta=eta, samples=samples, nrows=n)

    for steps in steps_list:
        average_nexperiments(3, t, eta=eta, steps=steps,
                             samples=samples, D=D, Q=Q, Qtest=Qtest)
