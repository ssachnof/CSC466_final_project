# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import sys
from importer import Importer
from InduceC45 import DecisionTree
from classifier import Classify
import json
import string

DRY_RUN = "--dry-run"
THRESH = .3

def main():
	path = "data/yellow-small.csv"
	dry_run = False

	# parse tags
	if len(sys.argv) == 2:
		if sys.argv[1] == DRY_RUN:
			dry_run = True
		else:
			path = sys.argv[1]
	if len(sys.argv) == 3 and sys.argv[2] == DRY_RUN:
		dry_run =  True
		path = sys.argv[1]


	# parse input
	im = Importer(path)
	im.parse_data()

	# build tree
	tree = DecisionTree(im)
	tree.build_tree(THRESH)
	print(tree.tree_to_json())

	# classify data
	classify = Classify(path, tree.tree)
	classify.check_data(dry_run)
		


if __name__ == '__main__':
	main()			


