from decision_tree import DecisionTree
from sklearn.utils import resample

class RandomForest:
	def __init__(self, n_trees=10, max_depth=None):
		self.max_depth = max_depth
		self.y_pred_all = []
		self.n_trees = n_trees

	def train(self, X, y, X_test):
		""" Create subsamples of the original dataset with replacement
		Subsamples will be of same size of the original dataset
		resample uses random. So each tree gets a different subsample
		Construct number of trees and get all their individual predictions """
		for i in range(self.n_trees):
			X2, y2 = resample(X, y)
			clf = DecisionTree(max_depth=self.max_depth)
			clf.train(X2, y2)
			y_pred = clf.predict(X_test)
			self.y_pred_all.append(y_pred)

	def predict(self):
		""" Consider predictions from all the trees.
		For a given sample, take a majority vote for prediction """
		y_pred = []
		for i in range(len(self.y_pred_all[0])):
			count1, count2, count3 = 0, 0, 0
			for j in range(len(self.y_pred_all)):
				if(self.y_pred_all[j][i] == 0):
					count1 += 1
				elif(self.y_pred_all[j][i] == 1):
					count2 += 1
				else:
					count3 += 1

			maxi = max(count1, count2, count3)
			
			if(count1 == maxi):
				y_pred.append(0)
			elif(count2 == maxi):
				y_pred.append(1)
			else:
				y_pred.append(2)

		return y_pred