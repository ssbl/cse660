import bitarray
from pickle import load, dump
from pprint import pprint
from sys import argv

PICKLE_FILE = 'fdb.p'

if __name__ == '__main__':
    argc = len(argv)
    if argc != 3:
        print('usage: extract_data <rows> <columns>')
        exit(1)

    rows, cols = int(argv[1]), int(argv[2])
    final_db = load(open(PICKLE_FILE, 'rb'))

    cut_db = [bitarray.bitarray(cols) for _ in range(rows)]

    for i in range(rows):
        for c in range(cols):
            cut_db[i][c] = final_db[i][c]

    with open('cdb_{}_{}.p'.format(rows, cols), 'wb') as cdb:
        dump(cut_db, cdb)

    pprint(cut_db)
