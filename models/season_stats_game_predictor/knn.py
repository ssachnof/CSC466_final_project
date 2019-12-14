

# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import sys
from importer import Importer
from InduceC45 import DecisionTree
import json
import string
import validation
import math

def main():
	path = "data/iris.data.csv"
	k = 5
	# parse tags
	if len(sys.argv) >= 3:
		path = sys.argv[1]
		k = int(sys.argv[2])
	
	else:
		print("Usage: python3 knn.py <path> <k>")
		#return

	validation.validation(path, 10, "KNN", k=k, hot=True)

	'''# build tree
	tree = DecisionTree(im)
	tree.build_tree(THRESH)
	print(tree.tree_to_json())

	# classify data
	classify = Classify(path, tree.tree)
	classify.check_data()'''



def knn(data, val, k, class_variable):
	neigh = []
	if(len(data.index) < k):
		print("There aren't k data points")
		neigh = data[class_variable].unique()
		return pluralityOfList(neigh)


	for idx, row in data.iterrows():
		if(row[class_variable] == None):
			continue



		temp_dist = num_distance(row, val, class_variable)

		#else:
		#	temp_dist = 10

		if len(neigh) < k:
			neigh += [(row[class_variable], temp_dist)]

		else:
			i, dist = find_worst_neigh(neigh)
			if(temp_dist < dist):
				neigh[i] = (row[class_variable], temp_dist)

	return pluralityOfList(neigh)


def find_worst_neigh(l):
	maxDist = None
	idx = None
	for i in range(len(l)):
		#print(l, i, l[i])
		curDist = l[i][1]
		if idx == None:
			maxDist = curDist
			idx = i

		else: 
			if curDist > maxDist:
				maxDist = curDist
				idx = i

	return idx, maxDist



def num_distance(t1, t2, var):
	sum = 0
	#print(t1, t2)
	temp1 = t1.drop(var)
	temp2 = t2.drop(var)
	if(len(temp1)!=len(temp2)):
		print("lengths don't match -> temp1: ", len(temp1), temp1, " temp2: ", len(temp2), temp2)
	
	#print(temp1, temp2)
	for i in range(len(temp1)):
		if(temp1[i] == '?') or (temp2[i] == "?"):
			#print("found missing value")
			continue
		#print("t1[i]: ", t1[i]," t2[1]: ", t2[i])
		sum += math.pow(float(temp1[i]) - float(temp2[i]), 2)

	return math.sqrt(abs(sum))

def pluralityOfList(l): 
    return max(set(l), key = l.count)[0]


if __name__ == '__main__':
	main()


