import numpy as np
import random
import itertools

def create_dataset(n):
    assert n >= 0
    return np.random.random_integers(0, 1, n)

def sum_query(D, rows):
    return np.sum([D[i] for i in rows])

def sum_query_noisy(D, rows, E):
    return sum_query(D, rows) + random.uniform(0,E)

def subset(my_list):
	result = []
	for i in range(1,len(my_list)+1):
		result += list(itertools.combinations(my_list,i))
	return result

def binary_set(n):
	return list(itertools.product([0,1], repeat=n))

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def attack(D,E):
	n = len(D)
	noisy_query_result = {}
	for i in subset(range(0,n)):
		nq = sum_query_noisy(D,i,E)
		noisy_query_result[i] = nq
	# print (noisy_query_result)

	D_dash = binary_set(n)
	print("D_dash")
	# print(D_dash)
	print("attack")
	for d in D_dash:
		found = True
		for i in subset(range(0,n)):
			sq = sum_query(d, i)
			nqr = noisy_query_result[i]
			# print(d,i,abs(sq-nqr), sq,nqr)
			if abs(sq-nqr) > E:
				found = False
				break
		if found:
			return d


if __name__ == '__main__':
    D = create_dataset(12)
    # D = [1,1,1,1,1,1,1,1,1,1]
    E = 3
    print(D)
    # print(sum_query(D, [3, 5, 9]))
    # print(sum_query_noisy(D,[3,5,9],1))
    # print(subset([1,2,3,4]))
    # print(binary_set(4))

    for i in range(100):
    	D = create_dataset(12)
    	D_dash =  attack(D,E)
    	hamDist = hamming2(D_dash,D)
    	print(D,D_dash,hamDist)