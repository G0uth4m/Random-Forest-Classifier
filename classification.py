import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from random_forest import RandomForest

# Importing the dataset
dataset = pd.read_csv('GALEX_data-extended-feats.csv')

# Setting independent and dependent variables
X = dataset.iloc[:, 1:24].values
y = dataset.iloc[:, 0].values

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 211)

# Training the classifier on the training set
clf2 = RandomForest(n_trees=2, max_depth=7)
clf2.train(X_train, y_train, X_test)

# Predicting test set samples
y_pred2 = clf2.predict()

# Measuring performance of the classifier
print(classification_report(y_test, y_pred2))