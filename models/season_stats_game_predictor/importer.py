# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Importer:
	def __init__(self, file):
		# field we are looking for
		self.class_variable = None
		# dictionary [name : # of options]
		self.fields = {}
		self.file = file
		self.data_frame = None
		self.values = []

	def parse_data(self, hot=False):
		file = open(self.file, "r")

		titles = file.readline().strip().split(",") 
		valid = file.readline().strip().split(",")
		valid = [int(i) for i in valid]

		for i in range(len(titles)):
			self.fields[titles[i]] = valid[i]

		self.class_variable = file.readline().strip()

		data = file.read().split("\n")[:-1]
		table = []
		for line in data:
			line = line.strip().split(",")
			if len(line)>2:
				#print(line)
				table.append(line)

		self.data_frame = pd.DataFrame(data=table)
		self.data_frame.columns = titles
		if hot:
			cat_cols = []
			for key in self.fields.keys():
				if self.fields[key] > 0 and not key == self.class_variable:
					cat_cols+=[key]
					#print("column \'{}\' is a cat".format(key))

			#print(cat_cols)
			le = LabelEncoder()
			#ohe = OneHotEncoder()
			ct = ColumnTransformer([("onehot", OneHotEncoder(), cat_cols)])


			#categorical_cols = self.data_frame.columns[cat_mask].tolist()
			#self.data_frame[categorical_cols] = self.data_frame[categorical_cols].apply(lambda col: le.fit_transform(col))

			#self.data_frame[key] = le.fit_transform(self.data_frame[key])
			#		print(self.data_frame[key])
			#self.data_frame = ct.fit_transform(self.data_frame)
			#self.data_frame = le.inverse_transform(self.data_frame)

			self.data_frame = pd.get_dummies(self.data_frame, columns = cat_cols)
			#print("transformed: ", self.data_frame)
			self.fields = None

		self.values = self.data_frame[self.class_variable].unique()
		return self

	def remove_invalid(self):
		for f in self.fields:
			if f < 0:
				self.fields.pop(f)

	def get_frame(self):
		return self.data_frame

	def get_fields(self):
		return self.fields

	def get_variable(self):
		return self.class_variable

	def get_values(self):
		return self.values

	def print_info(self):
		print("Looking for '{}'".format(self.class_variable))
		print("Fields are: \n{}".format(self.fields))
		print("Values are: \n{}".format(self.values))
		print(self.data_frame)

#im = Importer("data/adult-stretch.csv")
#im.parse_data()
#im.print_info()
