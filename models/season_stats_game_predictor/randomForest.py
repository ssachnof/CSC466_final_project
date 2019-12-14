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
import classifier
from statistics import mode

THRESH = 0.4

def main():
	path = "data/iris.data.csv"
	m = 3
	k = 10
	N = 5

	if len(sys.argv) == 5:
		path = sys.argv[1]
		m = int(sys.argv[2])
		k = int(sys.argv[3])
		N = int(sys.argv[4])
	
	else:
		print("Usage: python3 randomForest.py <path> <NumAtributes> <NumDataPoints> <NumTrees>")
		#return

	# parse input
	im = Importer(path)
	im.parse_data()

	validation.validation(path, 10, "RF", m=m, k=k, N=N)

	'''# build tree
	tree = DecisionTree(im)
	tree.build_tree(THRESH)
	print(tree.tree_to_json())

	# classify data
	classify = Classify(path, tree.tree)
	classify.check_data()'''


"m is the number of attributes per tree, k is the number of datapoints per tree and N is the number of trees to build"
def randomForest(data, val, class_variable, m, k, N):
	subdatasets = randomDatasets(data, val, class_variable, m, k, N)
	trees = []
	for subdata in subdatasets:
		trees += [DecisionTree.C45(subdata, dict.fromkeys(list(subdata)), class_variable, THRESH)]

	return trees



"m is the number of attributes per tree, k is the number of datapoints per tree and N is the number of trees to build"
def randomDatasets(data, val, class_variable, m, k, N):
	subdatasets = []
	for i in range(N):
		subdata = data.sample(k,replace=True)
		subsubdata = subdata.sample(m, axis = 1, replace=False)
		#print(subsubdata)
		subsubdata[class_variable] = subdata[class_variable]
		#print(subdata)
		#print(subsubdata)
		subdatasets += [subsubdata]
		#subdata

	return subdatasets


def classifyFromForest(trees, val):
	predictions = [] 
	for tree in trees:
		predictions += [classifier.Classify.classifyC45(tree, val)]

	#print("predictions: ", predictions)

	return pluralityOfList(predictions)
	



def knn(data, val, k, class_variable, catagorical=False):
	neigh = []
	if(len(data.index) < k):
		print("There aren't k data points")
		neigh = data[class_variable].unique()
		return pluralityOfList(neigh)


	for idx, row in data.iterrows():
		if(row[class_variable] == None):
			continue
		if(not catagorical):
			temp_dist = num_distance(row, val, class_variable)

		else:
			temp_dist = 10

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
	for i in range(len(temp1)):
		#print("t1[i]: ", t1[i]," t2[1]: ", t2[i])
		sum += math.pow(float(temp1[i]) - float(temp2[i]), 2)

	return math.sqrt(abs(sum))

def pluralityOfList(l): 
    return max(set(l), key=l.count)


if __name__ == '__main__':
	main()


