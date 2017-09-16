import random

def cointoss():
    return random.randint(0, 1)

def almost_random(d):
    if cointoss():
        return d
    else:
        return cointoss()

if __name__ == '__main__':
    iterations = 100000
    d, d_ = 1, 0

    pd = [almost_random(d) for _ in range(iterations)]
    pd_ = [almost_random(d_) for _ in range(iterations)]

    pd0 = [p for p in pd if p == 0]
    pd1 = [p for p in pd if p == 1]

    pd_0 = [p for p in pd_ if p == 0]
    pd_1 = [p for p in pd_ if p == 1]

    print(len(pd0) / len(pd_0), len(pd1) / len(pd_1))
