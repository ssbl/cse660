import math
import random
import sys
import time

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

def rr_sum(D, q, e):
    q_result = q(D)
    S = biased_predicate(q_result, e)

    return sum(S)/len(S)

def round_float(x):
    string = '{:.2f}'.format(x)
    return float(string)

def accuracy(e=0.5, n=10000, beta=0.05):
    # constants
    p = (1 + math.exp(e)) / (math.exp(e) - 1)
    q = 1 / (1 + math.exp(e))

    alpha = p * math.sqrt(math.log(2 / beta) / (2*n))
    print('alpha'+str(alpha))

    k, D, D_ = create_dataset_pair(n)
    nruns = range(1, 10000)

    dq = no_noise_query(D, predicate)
    D_results = [rr_sum(D, predicate, e) for _ in nruns]
    abs_error = [abs((p*(d-q))-dq) for d in D_results]
    error = [(p*(d-q))-dq for d in D_results]
    outside_alpha = []
    for e in abs_error:
        if e >= alpha:
            outside_alpha.append(1)
        else:
            outside_alpha.append(0)
    points_outside_alpha = sum(outside_alpha)
    return error, points_outside_alpha, alpha
    # i = plt.figure(3)
    # plt.axhline(0, color='g')
    # alpha_line = plt.axhline(alpha, color='r')
    # alpha_line2 = plt.axhline(-alpha, color='r')
    # plt.plot(error, 'go')
    # plt.xlabel('Nth run');
    # plt.ylabel('Error');
    # plt.legend([alpha_line],['alpha = '+'{:.2f}'.format(alpha)])
    print('points outside alpha = '+str(points_outside_alpha))
    # # plt.show()
    # i.show()

def privacy_loss():
    n = 100
    e = 0.5

    k, D, D_ = create_dataset_pair(n)
    iterations = range(1, 10000)

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

    print(len(losses))
    f = plt.figure(1)
    plt.plot(range(len(losses)), losses)
    legend_patch = mpatches.Patch(color='blue', label='privacy loss for epsilon = '+str(e))
    plt.legend(handles=[legend_patch])
    f.show()
    # plt.show()

def privacy_loss_histogram():
    n = 100
    e = 0.5
    k, D, D_ = create_dataset_pair(n)
    nruns = range(1, 1000)
    sum_so_far = 0
    result, result_ = [], []
    for i in nruns:
        x = round_float(rr_sum(D, predicate, e))
        y = round_float(rr_sum(D_, predicate, e))
        sum_so_far += x

        result.append(x)
        result_.append(y)

    g = plt.figure(2)
    h1 = plt.hist(result, bins=25)
    h2 = plt.hist(result_, bins=25)

    # plt.legend([h1, h2], ["RR(D)", "RR(D')"])
    # plt.plot(nruns, result)

    g.show()
    # plt.show()
    

if __name__ == '__main__':
    # privacy_loss()
    # privacy_loss_histogram()
    accuracy()
    input()
