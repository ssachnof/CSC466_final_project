# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
from importer import Importer
import math
import sys
import json
import string



def main():
	THRESH = 0.2
	path = '../../ryan_data/season_data_2018.csv'

	# parse tags
	if len(sys.argv) == 2:
		path = sys.argv[1]
		if len(sys.argv) == 3:
			THRESH = float(sys.argv[2])

	# parse input
	im = Importer(path)
	im.parse_data()

	# build tree
	tree = DecisionTree(im)
	tree.build_tree(THRESH)
	print(path)
	print()
	print(tree.tree_to_json())



class DecisionTree:
	def __init__(self, importer):
		self.importer = importer
		self.tree = None

	def build_tree(self, thresh=0.0):
		self.tree = DecisionTree.C45(self.importer.get_frame(), self.importer.get_fields(), self.importer.get_variable(), thresh)

	def tree_to_json(self):
		if(self.tree == None):
			return "The tree is empty."
		json_string = self.tree.node_to_json()
		return json_string

		#with open('test_file.json', 'w') as file:
    	#	json.dump(my_json_string, file)

	@staticmethod
	def C45(data, attrs, class_variable, thresh=0):
		#if all remaining data has the same class label
		#class_variable = self.importer.class_variable
		dom_attrs = data[class_variable].unique()

		if len(dom_attrs) == 1:
			# print("pure set")
			#print(dom_attrs[0])
			node = Node(dom_attrs[0], None, 1)
			return node

		elif len(attrs) == 1:
			# print("not more attrs")
			label, prob = DecisionTree.find_most_frequent_label(data, class_variable)
			# print(label[0])
			node = Node(label[0], None, prob)
			return node

		else:
			#returns the best splitting attribute or none if the info gain is less than thresh
			splitAttr, split_val = selectSplittingAttribute(attrs, data, class_variable, thresh)
			
			# attrs.remove(splitAttr)
			#none exceed thresh, choose plurality label
			if(splitAttr == None):
				# print("not enough gain")
				label, prob = DecisionTree.find_most_frequent_label(data, class_variable)
				node = Node(label, None, prob)
				return node

			else:
				is_numeric = attrs[splitAttr] == 0

				if not is_numeric:
					del attrs[splitAttr]
				#tree construction
				label, prob = DecisionTree.find_most_frequent_label(data, class_variable)
				node = Node(splitAttr, None, prob, label)

				if is_numeric:
					# data[splitAttr] = data[splitAttr].replace('?', 0)
					low_data = data[data[splitAttr] <= split_val]
					high_data = data[data[splitAttr] > split_val]

					if not low_data.empty:
						name = "<=" + str(split_val)
						edge = Edge(name, DecisionTree.C45(low_data, attrs, class_variable, thresh))
						node.add_edge(edge)
					if not high_data.empty:
						name = ">" + str(split_val)
						edge = Edge(name, DecisionTree.C45(high_data, attrs, class_variable, thresh))
						node.add_edge(edge)
					return node
				else:
					values = data[splitAttr].unique()
					for value in values:
						#data[data[a] == v]
						#print("value in values: ", value)
						splitdata = data[data[splitAttr] == value]
						if not data.empty:
							edge = Edge(value, DecisionTree.C45(splitdata, attrs, class_variable, thresh))
							node.add_edge(edge)
					return node
	
	@staticmethod
	def find_most_frequent_label(data, var):
		#print("data[var]: ", data[var])
		#print("mode: ", data[var].mode())
		if data[var].empty:
			return "None", 0
		# print("mode", data[var].mode())
		try:
			mode = data[var].mode()[0]
			prob = data.loc[data[var] == mode].count()/data[var].count()
			return mode, prob[1]
		except:
			return "None", 0

class Node:
	def __init__(self, label, edges, prob, plurality=None):
		self.label = label
		self.edges = edges
		self.prob = prob
		self.plurality = plurality
		#if self.is_leaf:
			#print(self.label)

	def add_edge(self, edge):
		if self.edges == None:
			self.edges = [edge]

		else:
			self.edges += [edge]

	def node_to_json(self):
		#print(type(self.label), print(type(self.edges)), print(self.prob))
		if not self.is_leaf():
			string = "node: { \n\tvar: " + self.label + "\n\tplurality: " + self.plurality + "\n\tprobability: " + str(self.prob) +\
			",\n\tedges: \n\t[ \n\t" + addindent(self.edges_to_json(),1) + "\n\t]\n}"
		
		else:
			string = "leaf: {decision: " + self.label + ",\n\tp: " + str(self.prob)+"}"
		#print(type(string))
		return string
	def edges_to_json(self):
		string = ""
		if self.edges == None:
			return string
		
		for edge in self.edges:
			string += addindent(edge.edge_to_json(), 1)

		return string

	def is_leaf(self):
		if(self.edges == None):
			return True

		return False

class Edge:
	def __init__(self, val, tree):
		self.val = val
		self.tree = tree

	def edge_to_json(self):
		nextType = "node"
		if(self.tree.is_leaf()):
			nextType = "leaf"
		string = "\n\t{\n\t\tvalue: " + self.val + ",\n\t" + addindent(self.tree.node_to_json(), 1) + "\n\t}"
		return string


def addindent(s, numTabs):
    s = s.split('\n')
    s = [(numTabs * '\t') + line for line in s]
    s = '\n'.join(s)
    return s

def entropy(data, var):
	occurs = data[var].value_counts()
	total = data.shape[0]
	entropy = 0.0

	for value in occurs:
		pi = (float(value)/float(total))
		entropy += -pi*math.log(pi, 2)

	return entropy

def entropy_split(data, var, dt):
	occurs = data[var].value_counts()
	total = data.shape[0]
	entropy = 0.0

	for value in occurs:
		pi = (float(value)/float(total))
		entropy += -pi*math.log(pi, 2)

	return entropy * (float(total) / float(dt))


def selectSplittingAttribute(attrs, data, var, thresh):
	#del atts[var]
	atts = attrs.keys()
	total_ent = entropy(data, var)
	data_total = data.shape[0]
	max_gain = 0
	max_att = None
	max_split = None

	for a in atts:
		if a == var:
			continue
		is_numeric = attrs[a] == 0
		att_ent = 0.0

		if is_numeric:
			att_ent, split = find_best_split(data, var, a, data_total)
			att_gain = total_ent - att_ent
			if att_gain > max_gain:
				max_gain = att_gain
				max_att = a
				max_split = split
		else:
			values = data[a].unique()

			for v in values:
				split = data[data[a] == v]
				att_ent += entropy_split(split, var, data_total)

			att_gain = total_ent - att_ent
			if att_gain > max_gain:
				max_gain = att_gain
				max_att = a

	if max_gain < thresh:
		return None, None

	return max_att, max_split

def find_best_split(data, var, a, dt):
	min_ent = 2
	min_split = None
	l = h = -1

	for row in data.iterrows():
		row = row[1]
		split = row[a]

		low = data[data[a] <= split]
		high = data[data[a] > split]
		if low.shape[0] == 0 or high.shape[0] == 0:
			continue

		ent = entropy_split(low, var, dt) + entropy_split(high, var, dt)
		if min_ent > ent:
			min_ent = ent
			min_split = split
			l = low.shape[0]
			h = high.shape[0]

	return min_ent, min_split
	


# im = Importer("data/adult-stretch.csv")
# im.parse_data()

# f = im.get_fields()
# d = selectSplittingAttribute(f, im.get_frame(), im.get_variable(), .2)
# print(d)
# print(f)

if __name__ == '__main__':
	main()