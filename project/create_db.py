#get all the file names

#create  a list of size 2649429

#for each file
	#movie_id = readLine()
	#for each line
	  	#gett the first number is uid
	  	#db[uid].append(movie_id)


#final_db = []

#for each list in db
	#create a list of 17770 size
	#for each element in list
		#list[element] = 1
	#final_db.append(list)

# from os import walk
import bitarray
import sys
# import glob
import pickle

# files=glob.glob('mv_*')

# print(files)
# exit(1)

# db = [[] for _ in range(2649429 + 1)]
# current_file_number = 1
# for f in files:
# 	print(current_file_number)
# 	current_file_number +=1
# 	with open(f) as movie_file:
# 		movie_id = int(movie_file.readline()[:-2])
# 		for line in movie_file:
# 			uid = int(line.split(',')[0])
# 			db[uid].append(movie_id)

# pickle.dump(db, open("db.p","wb"))

print("reading pickle", flush=True)
db = pickle.load(open("db.p","rb"))
print("done")

final_db = [bitarray.bitarray(17770) for _ in range(480189)]
for bitarr in final_db:
	bitarr.setall(False)

print("Starting to create db", flush=True)
row_counter = 0
for i, li in zip(range(len(db)), db):
	if not li:
                continue
	print(i)
	exit(1)
	print(str(row_counter).rjust(6, '0'), end='\r', flush=True)
	for index in li:
		final_db[row_counter][index-1] = True
	row_counter += 1

# pickle.dump(final_db, open('fdb.p', 'wb'))
# final_db = pickle.load(open('fdb.p', 'rb'))

print()
# print(final_db)
print(sys.getsizeof(final_db))
print(final_db[0])
