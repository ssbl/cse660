import json

import matplotlib.pyplot as plt

def parse(path):
    epsilon = []
    max_error = []
    avg_error = []

    with open(path) as log:
        for line in log:
            data = json.loads(line)
            max_error.append(data['max_error'])
            avg_error.append(data['avg_error'])
            T = data['steps']
            s = data['samples']
            eta = data['eta']
            n = data['n']
            e = T * (T - 1) * s * eta / n
            epsilon.append(e)

    return epsilon, max_error, avg_error

def plot_error(epsilon, error):
    plt.plot(epsilon, error, 'go')
    plt.show()

if __name__ == '__main__':
    epsilon, max_error, avg_error = parse('log.json')
    plot_error(epsilon, avg_error)
