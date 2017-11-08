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

def linear_query(D, y):
    result = [0.5*y[0] for d in D if d == y]
    return sum(result) / len(D)

def gaussmech(D, qD, e, delta, l2sensitivity):
    # b = 1.0 / (len(D) * e)
    b = (2 * np.log(1.25 / delta) * l2sensitivity**2) / e**2
    return qD + np.random.laplace(scale=b)

def round_float(x, n):
    fmt = '{:.' + str(n) +  'f}'
    string = fmt.format(x)
    return float(string)

def privacy_loss(query=counting_query):
    e = 0.5
    delta = np.exp(-20)
    n = 10000
    iterations = 10000
    D, D_ = create_dataset_pair(n)
    qD = query(D, (1, 1, 0))
    qD_ = query(D_, (1, 1, 0))
    result_, result = [], []
    for i in range(iterations):
        result.append(gaussmech(D, qD, e, delta, 1.0))
        result_.append(gaussmech(D_, qD_, e, delta, 1.0))

    min_bound = min(min(result), min(result_))
    max_bound = max(max(result), max(result_))
    result.append(min_bound)
    result.append(max_bound)
    result_.append(min_bound)
    result_.append(max_bound)

    bins = 30
    counts, edges = np.histogram(result, bins=bins)
    counts_, edges_ = np.histogram(result_, bins=bins)
    print(counts)
    print(counts_)
    ratios = [1.0*a/b for a,b in zip(counts, counts_)]
    assert len(counts) == len(ratios)
    print(ratios)

    i = plt.figure(0)
    plt.axhline(np.exp(e), color='r', label='ε')
    plt.axhline(np.exp(-e), color='r', label='-ε')
    plt.legend()
    plt.plot(ratios)
    i.show()
    g = plt.figure(1)
    plt.hist(result, bins=50, label='min bound')
    plt.hist(result_, bins=50, label='max bound')
    plt.legend()
    g.show()
    input()

def accuracy(query=counting_query):
    e = 0.5
    delta = np.exp(-20)
    n = 10000
    beta = 0.05
    alpha = (1.0 / (n * e)) * np.log(1 / beta)
    iterations = 10000
    print('alpha = {}'.format(alpha))

    D, D_ = create_dataset_pair(n)
    qD = query(D, (1, 1, 0))

    result = []
    for i in range(iterations):
        result.append(gaussmech(D, qD, e, delta, 1.0))

    error = [qD - r for r in result]
    abs_error = [abs(qD - r) for r in result]
    error_points = sum([1 if r >= alpha else 0 for r in abs_error])
    calculated_beta = 1.0 * error_points / n
    print(calculated_beta)
    i = plt.figure(3)
    plt.axhline(0, color='g')
    alpha_line = plt.axhline(alpha, color='r')
    alpha_line2 = plt.axhline(-alpha, color='r')
    plt.legend([alpha_line],['calculated β = {:.3f}'.format(calculated_beta)])
    plt.plot(error, 'go')
    i.show()
    input()

if __name__ == '__main__':
    privacy_loss(counting_query)
    # accuracy(counting_query)
    # privacy_loss(linear_query)
    # accuracy(linear_query)
