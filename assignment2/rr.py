import math
import random
import sys
import time

from collections import defaultdict

import matplotlib.pyplot as plt

def create_dataset_pair(n):
    D = [random.randint(1, 50) for _ in range(n)]

    k = random.randrange(n)
    D_ = D[:]

    while D_[k] == D[k]:
        D_[k] = random.randint(1, 50)

    return k, D, D_

def predicate(D):
    return [d % 2 for d in D]

def biased_flip(e):
    p = math.exp(e) / (1.0 + math.exp(e))
    if random.random() < p:
        return 1
    return 0

def biased_predicate(D, e):
    result = []

    for d in D:
        if biased_flip(e):
            result.append(d)
        else:
            result.append(0 if d else 1)
    return result

def no_noise_query(D, q):
    result = q(D)
    return sum(result) / len(result)

def rr(D, q, e):
    q_result = q(D)
    S = biased_predicate(q_result, e)

    return S

def round_float(x):
    string = '{:.2f}'.format(x)
    return float(string)

def accuracy():
    n = 100
    e = 0.1
    beta = 0.1

    # constants
    p = (1 + math.exp(e)) / (math.exp(e) - 1)
    q = 1 / (1 + math.exp(e))

    alpha = p * math.sqrt(math.log(2 / beta) / (2*n))
    print(alpha)

    k, D, D_ = create_dataset_pair(n)
    nruns = range(1, 1000)

    dq = no_noise_query(D, predicate)
    D_results = [rr(D, predicate, e) for _ in nruns]
    D__results = [rr(D_, predicate, e) for _ in nruns]

def privacy_loss():
    n = 10
    e = 0.1

    k, D, D_ = create_dataset_pair(n)
    iterations = range(1, 100000)

    losses = []
    q_count, q__count = 0, 0

    for i in iterations:
        S = rr(D, predicate, e)
        S_ = rr(D_, predicate, e)
        q = predicate(D)
        q_ = predicate(D_)
        if q[k] != q_[k]:
            if S[k] == q[k]:
                q_count += 1
            if S[k] == q_[k]:
                q__count += 1
        if q__count and q_count:
            result = math.log(1.0 * q_count / q__count)
            losses.append(result)

    plt.plot(range(len(losses)), losses)
    plt.show()

if __name__ == '__main__':
    privacy_loss()

'''
if __name__ == '__main__':
    n = 100
    e = 0.1
    # beta = 0.1

    # constants
    # p = (1 + math.exp(e)) / (math.exp(e) - 1)
    # q = 1 / (1 + math.exp(e))

    # alpha = p * math.sqrt(math.log(2 / beta) / (2*n))
    # print(alpha)

    D, D_ = create_dataset_pair(n)
    nruns = range(1, 1000)

    # dq = no_noise_query(D, predicate)
    # D_results = [rr(D, predicate, e) for _ in nruns]
    # D__results = [rr(D_, predicate, e) for _ in nruns]

    # print(len([d for d in D_results if abs(p*(d-q) - dq) >= alpha]) / n)
    # for d in D_results:
    #     if abs(p*(d-q) - dq) >= alpha:
    #         print(d)

    # counts = defaultdict(int)
    # counts_ = defaultdict(int)
    # for result in D_results:
    #     counts[result] += 1
    # for result in D__results:
    #     counts_[result] += 1

    # count = 0
    # sum_of_ratios = 0
    # losses = []
    # for k in counts:
    #     if counts_[k]:
    #         ratio = math.log(counts[k] / counts_[k])
    #         count += 1
    #         sum_of_ratios += ratio
    #         losses.append(ratio)

    # print(max(losses))
    # plt.axhline(sum_of_ratios / count)
    # plt.plot(losses)
    # plt.show()

    # ================================================
    sum_so_far = 0
    result, result_ = [], []
    for i in nruns:
        x = round_float(rr(D, predicate, e))
        y = round_float(rr(D_, predicate, e))
        sum_so_far += x

        result.append(x)
        result_.append(y)

    print(math.exp(e))
    print(math.exp(-e))
    h1 = plt.hist(result, bins=25)
    h2 = plt.hist(result_, bins=25)
    # plt.legend([h1, h2], ["RR(D)", "RR(D')"])
    # plt.plot(nruns, result)
    plt.show()
'''
