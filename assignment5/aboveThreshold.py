import numpy as np
import matplotlib.pyplot as plt

def create_dataset_pair(n,nbits):
    columns = []
    for i in range(nbits):
        columns.append(np.random.randint(2, size=n))

    D = list(zip(*columns))
    return D

def counting_query(D, y):
    result = [1 for d in D if d == y]
    return sum(result)


def above_threshold(D, query_list, T, epsilon):
    result = []
    T_noise = np.random.laplace(scale=2/epsilon)
    T_ = T + T_noise
    for query in query_list:
        vi = np.random.laplace(scale=4/epsilon)
        if  counting_query(D, query) + vi >= T_:
            result.append(True)
            break

        else:
            result.append(False)
    return result

def sigmaE(c,e):
    return 2.0 * c / e;

def multiple_above_threshold(D, query_list, T, epsilon, c):
    e1 = 8.0/9 * epsilon
    e2 = 2.0/9 * epsilon

    T_noise = np.random.laplace(scale=sigmaE(c,e1))
    count = 0
    result = []
    for query in query_list:
        vi = np.random.laplace(scale=2*sigmaE(c,e1))
        if counting_query(D,query)+vi >= T_noise:
            vi = np.random.laplace(scale=sigmaE(c,e2))
            ai = counting_query(D,query)+vi
            count += 1
            T_noise = np.random.laplace(scale=sigmaE(c,e1))
            result.append(ai)
        else:
            result.append(False)
        if count >= c:
            break

    return result;

def create_queries(n,nbits):
    a = np.arange(2**nbits)
    np.random.shuffle(a)
    return [tuple(map(int,bin(d)[2:].rjust(nbits,'0'))) for d in a[:n]]

if __name__ == '__main__':
    n = 1000
    nbits = 7
    kmax = 50
    D = create_dataset_pair(n,nbits)
    T = (n / (2**nbits -1)) + 1
    epsilon = 5
    beta = 0.2

    query_list = create_queries(kmax,nbits)
    query_list_result = [counting_query(D,q) for q in query_list]
    while(True):
        violations = 0
        nturns = 500
        for i in range(nturns):
            result = above_threshold(D, query_list, T , epsilon)
            # result = multiple_above_threshold(D, query_list, T , epsilon, 10)
            result_len = len(result)
            k = result_len
            alpha = 8*(np.log(k)-np.log(2/beta))/epsilon

            for j in range(result_len):
                if result[j] == False:
                    if query_list_result[j] > T + alpha:
                        violations +=1
                        break
                else:
                    if query_list_result[j] < T - alpha:
                        violations +=1
                        break

        print('calc beta', violations/nturns)
