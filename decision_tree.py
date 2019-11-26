import numpy as np

""" Creating a blue print for a node in the binary tree """
class TreeNode:
	def __init__(self, pred_class):
		self.pred_class = pred_class
		self.index = 0
		self.threshold = 0
		self.left = None
		self.right = None

class DecisionTree:
	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	def find_best_split(self, X, y):
		no_of_samples = y.size

		""" No. of samples belonging to different classes in the node """
		parent_node = []
		for i in range(self.no_of_classes):
			parent_node.append(np.sum(y == i))

		""" Calculating gini impurity of the current node """
		summation = 0
		for j in parent_node:
			summation += (j/no_of_samples)**2

		best_gini = 1 - summation

		index_best = None
		threshold_best = None

		for index in range(self.no_of_features):
			""" Select a feature and sort it in increasing order """
			thresholds, classes = zip(*sorted(zip(X[:, index], y)))
			
			""" Initially left child has no samples; right child has all samples from parent. """
			left_child = []
			for k in range(self.no_of_classes):
				left_child.append(0)

			right_child = parent_node.copy()

			# Iteratively find the best feature to be split and the best threshold
			""" Start transferring the sorted values of a particular feature one by one 
			from right child to left child """
			for i in range(1, no_of_samples):
				_class = classes[i - 1]
				
				left_child[_class] += 1
				right_child[_class] -= 1

				""" Find gini impurity of left child """
				summation1 = 0
				for x in range(self.no_of_classes):
					summation1 += (left_child[x]/i)**2
				gini_left_child = 1.0 - summation1
				
				""" Find gini impurity of right child """
				summation2 = 0
				for x in range(self.no_of_classes):
					summation2 += (right_child[x]/(no_of_samples - i))**2
				gini_right_child = 1.0 - summation2
				
				""" Find weighted gini impurity using left gini and right gini """
				gini = (i*gini_left_child + (no_of_samples - i)*gini_right_child) / no_of_samples

				if(thresholds[i] == thresholds[i - 1]):
					continue

				""" Check if the calculated gini impurity is better(lower) than the parent's
				gini impurity. """
				if(gini < best_gini):
					best_gini = gini
					index_best = index
					threshold_best = (thresholds[i-1] + thresholds[i]) / 2
		
		l = [index_best, threshold_best]			
		return l

	def createTree(self, X, y, depth=0):

		""" No. of samples belonging to different classes in the node """
		no_of_samples_per_class = []
		for i in range(self.no_of_classes):
			no_of_samples_per_class.append(np.sum(y == i))

		""" Find maximum number of samples that corresponds to same class """
		majority_class = np.argmax(no_of_samples_per_class)
		treeNode = TreeNode(pred_class = majority_class)

		""" Construct a binary tree recursively while choosing the best split and best threshold """
		if(depth < self.max_depth):
			l = self.find_best_split(X, y)
			index = l[0]
			threshold = l[1]

			if(index != None):
				left_indices = X[:, index] < threshold
				X_left = X[left_indices]
				y_left = y[left_indices]
				X_right = X[~left_indices]
				y_right = y[~left_indices]

				treeNode.index = index
				treeNode.threshold = threshold
				treeNode.left = self.createTree(X_left, y_left, depth+1)
				treeNode.right = self.createTree(X_right, y_right, depth+1)

		return treeNode

	
	def train(self, X, y):
		self.no_of_classes = len(set(y))
		self.no_of_features = X.shape[1]
		self.tree = self.createTree(X, y)

	def predict_row(self, row):
		treeNode = self.tree
		""" Traverse the tree until leaf node to find the class """
		while(treeNode.left):
			if(row[treeNode.index] < treeNode.threshold):
				treeNode = treeNode.left
			else:
				treeNode = treeNode.right
		return treeNode.pred_class				

	def predict(self, X):
		l = []
		for row in X:
			l.append(self.predict_row(row))
		return l