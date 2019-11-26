# Random-Forest-Classifier
An ensemble learning algorithm implemented from scratch in Python

## Problem Statement
To classify stellar objects into spectrometric classes based on photometric data. The classes are 0, 1, 2.

## Decision Trees
Decision trees can be used for regression (continuous real-valued output) or classification (categorical output).
A decision tree classifier is a binary tree where predictions are made by traversing the tree from root to leaf — at each node, we go left if a feature is less than a threshold, right otherwise.
Finally, each leaf is associated with a class, which is the output of the predictor.

## Gini Impurity
Decision trees use the concept of Gini impurity to describe how homogeneous or “pure” a node is.
A node is pure (G = 0) if all its samples belong to the same class, while a node with many samples from many different classes will have a Gini closer to 1.

Gini impurity of n training samples split across k classes is defined as
![Gini](https://miro.medium.com/max/373/1*UNszwSYfUJFHtfC0jvBKsw@2x.png)
where pk is the fraction of samples belonging to class k.

## Working
Each node is split so that the average of the Gini of the children weighted by their size is minimized.
The recursion stops when the maximum depth, a hyperparameter, is reached, or when no split can lead to two children purer than their parent.
In the given dataset, we have continuous independent variables, so we use a technique where we sort the values for a given feature and consider all midpoints between two adjacent values as possible thresholds.
To find the optimal feature and threshold such that the Gini impurity is minimized, we try all possible splits and compute the resulting Gini impurities.

## Random Forests
Decision trees have a common problem of overfitting.
So, we bring in ensemble learning. We build number of decision trees which are constructed using subsamples of the original dataset with replacement (same size as original dataset).
We consider the predictions of all the 100 trees. For each test sample, we take the majority vote and then predict for that particular test sample.
As a result, we were able to improve the accuracy in predictions and also precision and recall scores.

## Author
* **Goutham** - [G0uth4m](https://github.com/G0uth4m)
* **Monish Reddy** - [MONISHREDDYBS](https://github.com/MONISHREDDYBS)
