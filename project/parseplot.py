import json
from sys import argv, stderr

import matplotlib.pyplot as plt

def parse(pathlist):
    epsilon = []
    max_error = []
    avg_error = []

    for path in pathlist:
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

if __name__ == '__main__':
    argc = len(argv)
    if argc <= 1:
        print('usage: parseplot <path-to-result>...', file=stderr)
        exit(1)

    files = argv[1:]
    epsilon, max_error, avg_error = parse(files)

    max_error_plot = plt.figure(1)
    plt.plot(epsilon, max_error, '-go')
    plt.xlabel('epsilon')
    plt.ylabel('max error')

    avg_error_plot = plt.figure(2)
    plt.plot(epsilon, avg_error, '-go')
    plt.xlabel('epsilon')
    plt.ylabel('average error')
    max_error_plot.show()
    avg_error_plot.show()
    input()
