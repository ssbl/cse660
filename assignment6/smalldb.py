import numpy as np
import matplotlib.pyplot as plt
import itertools

def create_dataset_pair(n,nbits):
    columns = []
    for i in range(nbits):
        columns.append(np.random.randint(2, size=n))

    D = list(zip(*columns))
    return D

def create_queries(n,nbits):
    a = np.arange(2**nbits)
    np.random.shuffle(a)
    return [tuple(map(int,bin(d)[2:].rjust(nbits,'0'))) for d in a[:n]]

def counting_query(D, y):
    result = [1 for d in D if d == y]
    return sum(result) / len(D)

def create_universe(nbits):
	a = np.arange(2**nbits)
	return [tuple(map(int,bin(d)[2:].rjust(nbits,'0'))) for d in a]

def create_DI(X, m):
	return list(itertools.combinations(X,m))

def score(D, Di, Q):
	max = 0;
	for q in Q:
		result = abs(counting_query(D,q) - counting_query(Di,q))
		if result > max:
			max = result;
	return -max;

def generate_probabilities(D, u, R, epsilon, Q):
	delta_u = 1 / len(D)
	results = [np.exp((epsilon * u(D, r, Q)) / (2.0 * delta_u)) for r in R]
	total = sum(results)
	return [1.0 * x / total for x in results]

def accuracy(D,D_dash,Q):
	max = 0;
	for q in Q:
		result = abs(counting_query(D,q) - counting_query(D_dash,q))
		if result > max:
			max = result;
	return max


def lapmech(D, qD, e):
    b = 1.0 / (len(D) * e)
    return qD + np.random.laplace(scale=b)

def composition(D, query, ylist, budget):
    e = budget / len(ylist)
    query_results = []

    qDs =[query(D,y) for y in ylist]
    for qD in qDs:
        query_results.append(lapmech(D, qD, e))

    return query_results


if __name__ == '__main__':
	nbits = 5
	n = 32
	D = create_dataset_pair(n,nbits)
	Q = create_queries(25, nbits);
	qD = [counting_query(D,q) for q in Q]
	# print(qD)
	alpha = 0.33
	m = np.log(len(Q)) / alpha**2
	m = int(m)
	print("m",m)
	# exit(0)
	X = create_universe(nbits)
	DI = create_DI(X, m)
	epsilon = 0.1
	# print(DI)
	p = generate_probabilities(D, score, DI, epsilon, Q)
	
	D_index = np.random.choice(range(len(DI)), p=p)
	D_dash = DI[D_index]

	smallDBAcc = []
	composAcc = []

	for itern in range(1,25):

		Q_accuracy = create_queries(itern, nbits);
		acc = accuracy(D,D_dash,Q_accuracy)
		smallDBAcc.append(acc)
		print("SmallDB acc",acc)
		beta = 0.2

		new_alpha = ((16*np.log(len(X))*np.log(len(Q)) + 4 * np.log(1/beta)) / (epsilon * n)) ** (1/3)
		
		########### Composition
		comp_result = composition(D, counting_query, Q_accuracy, epsilon)
		qDs =[counting_query(D,q) for q in Q_accuracy]
		max = 0
		for i in range(len(comp_result)):
			if abs(qDs[i] - comp_result[i]) > max:
				max = abs(qDs[i] - comp_result[i])
		acc = max 
		composAcc.append(acc)
		print("composition accuracy", acc)

	smline = plt.plot(smallDBAcc, label='SmallDB')
	comline = plt.plot(composAcc, label='Composition')
	plt.legend()
	# plt.legend([smline],[ 'Composition Accuracy'])
	plt.xlabel('number of queries')
	plt.ylabel('accuracy')
	plt.show()

	

