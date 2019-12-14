# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import sys
from importer import Importer
from InduceC45 import DecisionTree
import json
import string

def main():
	THRESH = 0.2
	path = "data/yellow-small.csv"

	# parse tags
	if len(sys.argv) > 2:
		path = sys.argv[1]
		if len(sys.argv) == 3:
			THRESH = float(sys.argv[2])

	# parse input
	im = Importer(path)
	im.parse_data()

	# build tree
	tree = DecisionTree(im)
	tree.build_tree(THRESH)
	print(tree.tree_to_json())

	# classify data
	classify = Classify(path, tree.tree)
	classify.check_data()


class Classify:
	def __init__(self, file, tree):
		self.data = Importer(file).parse_data()
		self.training = True
		self.tree = tree
		self.records = {
			"num_records" : 0,
			"correct" : 0,
			"incorrect" : 0,
			"accuracy" : 0,
			"error_rate" : 0
		}

	def check_data(self, dry_run=True):
		if self.training:
			var = self.data.get_variable()

		for row in self.data.get_frame().iterrows():
			row = row[1]
			val = Classify.classify(self.tree, row)

			if self.training:
				correct = self.compare(val, row[var])
				self.record(correct)
			self.records["num_records"] += 1

			if not dry_run:
				self.output(row, val)
		self.print_stats()

	@staticmethod
	def classify(tree, row):
		var = tree.label
		if(tree.is_leaf()):
			return tree.label

		for edge in tree.edges:
			if row[var] == edge.val:
					return Classify.classify(edge.tree, row)

		#print("not found")
		return tree.plurality


	def compare(self, predicted, actual):
		return predicted == actual

	def record(self, correct):
		if correct:
			self.records["correct"] += 1
		else:
			self.records["incorrect"] += 1

	def output(self, data, val):
		print("{} -> {}".format(data, val))

	def print_stats(self):
		self.records["accuracy"] = float(self.records["correct"]) / float(self.records["num_records"])*100
		self.records["error_rate"] = float(self.records["incorrect"]) / float(self.records["num_records"])*100
		print("--------------------------")
		print("\tSTATISTICS")
		print("--------------------------")

		for val in self.records:
			print("{} : {}".format(val, self.records[val]))


if __name__ == '__main__':
	main()
