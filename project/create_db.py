""" This module generates dual query compatible dataset from netflix dataset"""
import bitarray
import sys
import glob
import pickle

NF_TOTAL_USERS = 2649429
NF_MOVIES = 17770
NF_USERS = 480189

def parse_netflix_files(directory, parsed_db_file):
    """ Parse all the movie files from the directory and create a dictionary
    mapping each user to the movies they have watched and store it as pickle
    file

    Args:
        directory: location of netflix movie files
        parsed_db_file: name of the pickle file to be stored
    """	
	files=glob.glob(sys.path.join(directory,'mv_*'))
	db = [[] for _ in range(NF_TOTAL_USERS + 1)]
	current_file_number = 1
	for f in files:
		print(current_file_number)
		current_file_number +=1
		with open(f) as movie_file:
			movie_id = int(movie_file.readline()[:-2])
			for line in movie_file:
				uid = int(line.split(',')[0])
				db[uid].append(movie_id)
	with open(parsed_db_file, "wb") as f:
		pickle.dump(db, f)

def create_netflix_dataset(parsed_db_file):
	""" Create dual query compatible netflix dataset using pickle file
	and store it as fdb.p

    Args:
        parsed_db_file: Pickle file containing a dictionary
    mapping each user to the movies they have watched
    """
	print("reading pickle", flush=True)
	with open(parsed_db_file,"rb") as f:
		db = pickle.load(f)
	print("done")

	final_db = [bitarray.bitarray(NF_MOVIES) for _ in range(NF_USERS)]
	for bitarr in final_db:
		bitarr.setall(False)

	print("Starting to create db", flush=True)
	row_counter = 0
	for i, li in zip(range(len(db)), db):
		if not li:
			continue
		print(str(row_counter).rjust(6, '0'), end='\r', flush=True)
		for index in li:
			final_db[row_counter][index-1] = True
		row_counter += 1
	print()
	with open('fdb.p', 'wb') as f:
		pickle.dump(final_db, f)


if __name__ == '__main__':
    argc = len(argv)
    if argc != 2:
        print('usage: create_db <path-to-netflix-directory>')
        exit(1)

    parsed_db_file = "db.p"
    directory = argv[1]
    if not Path(parsed_db_file).exists():
    	parse_netflix_files(directory, parsed_db_file)
    create_netflix_dataset(parsed_db_file)
