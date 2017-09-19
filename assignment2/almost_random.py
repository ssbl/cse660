import math
import random

import matplotlib.pyplot as plt

def cointoss():
    return random.randint(0, 1)

def almost_random(d):
    if cointoss():
        return d
    else:
        return cointoss()

if __name__ == '__main__':
    iterations = range(1, 100000)
    d, d_ = 1, 0

    result0, result1 = [], []
    pd, pd_ = [], []
    pd1, pd_1 = 0, 0

    for i in iterations:
        ar1 = almost_random(d)
        ar2 = almost_random(d_)

        pd1 += ar1
        pd_1 += ar2
        if pd_1 != 0 and pd1 != 0:
            result1.append(math.log(pd1 / pd_1))
        if i - pd_1 != 0 and i - pd1 != 0:
            result0.append(math.log((i - pd1) / (i - pd_1)))

    upper_bound = plt.axhline(math.log(3), color='r')
    lower_bound = plt.axhline(math.log(1/3), color='m')

    plt.plot(range(len(result1)), result1)
    plt.plot(range(len(result0)), result0)
    plt.legend((upper_bound, lower_bound),
               ['upper bound', 'lower bound'])
    plt.show()
