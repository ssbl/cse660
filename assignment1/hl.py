import itertools

import numpy as np


U = list('abcdefghij')

def create_histogram(limit=3):
    return np.random.random_integers(0, limit, len(U)).tolist()

def create_list(n=20):
    return [np.random.choice(U) for _ in range(n)]

def h2l(histogram):
    '''Convert a histogram to a (sorted) list.'''
    assert len(histogram) == len(U)
    iterator_list = [itertools.repeat(elem, count)
                     for (elem, count) in zip(U, histogram)]
    return list(itertools.chain.from_iterable(iterator_list))

def l2h(li):
    '''Convert a list to a histogram.'''
    return [len(list(g))-1 for _, g in itertools.groupby(sorted(U+li))]

if __name__ == '__main__':
    hlist = [create_histogram() for _ in range(3)]
    llist = [create_list() for _ in range(3)]

    print('h2l')
    for histogram in hlist:
        print('{} => {}'.format(histogram, h2l(histogram)))

    print('l2h')
    for li in llist:
        print('{} => {}'.format(li, l2h(li)))
