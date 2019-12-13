# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import sys
import numpy as np
from importer import Importer
from InduceC45 import DecisionTree
import classifier
import json
import string
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import randomForest

THRESH = 0.9

def main():
	global THRESH
	restrictions = None
	if len(sys.argv) >= 3:
		path = sys.argv[1]
		n = int(sys.argv[2])
		if len(sys.argv) == 4:
			THRESH = float(sys.argv[3])
		# if len(sys.argv) = 5:
		# 	THRESH = sys.argv[4]
	else:
		print("syntax: python3 validaiton.py <path> <n-fold (int)>")
		return

	# path = "data/yellow-small.csv"
	print("threshold: ",THRESH)
	validation(path, n)


def validation(path, n, classifier="RF", m=3, k=10, N=5, hot=False, dataset = "Data"):
	
	# parse input
	im = Importer(path)

	im.parse_data(hot=hot)

	# build tree
	#tree = DecisionTree(im)
	#tree.build_tree()
	#print(tree.tree_to_json())
	df = im.get_frame()

	if n >= 2:
		pass
	elif n == 0 or n == 0: 
		print("No Cross validation")
		return
	elif n == -1:
		print("all but one")
		n = df.shape[0]

	kf = KFold(n_splits=n, shuffle=True)

	labels = im.get_values()

	open("results.txt", "w")

	conf_list = []
	for trainidx, testidx in kf.split(df):
		train = df.iloc[trainidx]
		test =  df.iloc[testidx]

		if(classifier == "C45"):
			tree = DecisionTree.C45(train, im.get_fields().copy(), im.get_variable(), THRESH)
			conf = get_confusionC45(tree, test, im.get_variable(), labels)
			conf_list += [conf]

		elif(classifier == "KNN"):
			conf = get_confusionKNN(df, test, im.get_variable(), labels, k)
			conf_list += [conf]

		elif(classifier == "RF"):
			trees = randomForest.randomForest(df, test, im.get_variable(), m, k, N)
			conf = get_confusionRF(trees, test, im.get_variable(), labels, m, k, N)
			conf_list += [conf]
		
	
	print_output(conf_list, labels, n)

		#print(conf)


def get_confusionC45(tree, test, var, labels, out = "results.txt"):
	pred = []
	true = []
	# output = open("output/{}-c45.txt".format(out), "w")
	output = open(out, "a")
	output.write("C45 Classifications:\n\n")
	
	for idx, row in test.iterrows():
		val = classifier.Classify.classifyC45(tree, row)
		pred += [val]
		true += [row[var]]
		output.write("-- Row: -- \n{} -> \n-- Prediction --: {}\n\n".format(row, val))

	return confusion_matrix(true, pred, labels)


def get_confusionKNN(data, test, var, labels, k, out = "results.txt"):
	pred = []
	true = []
	output = open(out, "a")
	output.write("KNN Classifications:\n\n")
	
	for idx, row in test.iterrows():
		val = classifier.Classify.classifyKNN(data, row, var, k)
		pred += [val]
		if(val == "None"):
			true += [val]
		else:
			true += [row[var]]
		output.write("-- Row: -- \n{} -> \n-- Prediction --: {}\n\n".format(str(row), val))

	#print("pred: ", pred)
	#print("true: ", true)

	return confusion_matrix(true, pred, labels)

def get_confusionRF(trees, test, var, labels, m, k, n, out = "results.txt"):
	pred = []
	true = []
	output = open(out, "a")
	output.write("Random Forest Classifications:\n\n")
	
	for idx, row in test.iterrows():
		val = classifier.Classify.classifyRF(trees, row, var)
		pred += [val]
		#print("pred: ", val, "true: ", row[var])
		if(val == "None"):
			true += [val]
		else:
			true += [row[var]]
		output.write("-- Row: -- \n{} -> \n-- Prediction --: {}\n\n".format(row, val))

		#write(out, str(row)+"  ---  prediction: " + str(val))

	#print("pred: ", pred)
	#print("true: ", true)

	return confusion_matrix(true, pred, labels)

def add_confusions(conf_list):
	#print(conf_list)
	if len(conf_list) == 0:
		return None

	overall_conf = conf_list[0]
	for i in range(1, len(conf_list)):
		#print("adding conf: \n\t", overall_conf, "\n\t+\n\t",conf_list[i])
		overall_conf = np.add(overall_conf, conf_list[i])
		

	return overall_conf

def print_confusion(conf, labels):
	#print(type(conf))
	print("  ")
	for label in labels:
		print(label)

	print()
	for i in range(len(labels)):
		print(labels[i], conf[i])

def precision_from_conf(conf, labels):
	correct = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			#print("conf: ", conf)
			count = conf[i][j]
			if(not binary or (binary and j==0)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==0 and j==0):
				correct += count

	if total == 0:
		return 0

	return correct/total

def recall_from_conf(conf, labels):
	correct = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			if(not binary or (binary and i==0)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==0 and j==0):
				correct += count

	if total == 0:
		return 0

	return correct/total

def pf_from_conf(conf, labels):
	incorrect = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			if(not binary or (binary and i==1)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==1 and j==0):
				incorrect += count
	if total == 0:
		return 0

	return incorrect/total

def fmes_from_stats(precision, recall):
	numer = 2 * precision * recall
	denom = precision + recall
	fmes = numer / denom
	return fmes

def accuracy_from_conf(conf, labels):
	correct = 0
	total = 0

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			total += conf[i][j]

			if (i==j):
				correct += count

	if total == 0:
		return 0

	return correct/total

def print_output(conf_list, labels, n):
	print("--------------------------")
	print("\tVALIDATION")
	print("--------------------------")
	print("Folds:  {}\n".format(n))
	overall_conf = add_confusions(conf_list)
	#print("overall_conf", overall_conf)
	print("Confusion Matrix: ")
	print_confusion(overall_conf, labels)
	'''precision = precision_from_conf(overall_conf, labels)
	print("Precision: ", precision)
	recall = recall_from_conf(overall_conf, labels)
	print("Recall: ", recall)
	pf = pf_from_conf(overall_conf, labels)
	print("pf: ", pf)
	fmes = fmes_from_stats(precision, recall)
	print("F-measure: ", fmes)
	'''
	overall_accuracy = accuracy_from_conf(overall_conf, labels)
	print("Overall Accuracy: ", overall_accuracy)
	acc_list = []

	for conf in conf_list:
		#print(conf)
		#print()
		acc_list += [accuracy_from_conf(conf, labels)]

	average_accuracy = np.mean(acc_list)
	print("Average Accuracy: ",  average_accuracy)
	#print(acc_list)
	print("Overall Error Rate: ", 1-overall_accuracy)
	print("Average Error Rate: ", 1-average_accuracy)

if __name__ == '__main__':
	main()







