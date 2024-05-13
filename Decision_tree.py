#!/bin/python3

import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, Y):
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, Y)

    def _grow_tree(self, X, Y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(Y))

        # Check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        # Find the best split
        best_feature, best_thresh = self._best_split(X, Y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, Y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                # Calculate the information gain
                gain = self._information_gain(Y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def _information_gain(self, Y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(Y)
        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted avg. entropy of children
        n = len(Y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, Y):
        hist = np.bincount(Y)
        ps = hist / len(Y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, Y):
        counter = Counter(Y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Load breast cancer dataset
data = datasets.load_breast_cancer()
X, Y = data.data, data.target

# Split dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Create and fit DecisionTree classifier
clf = DecisionTree()
clf.fit(X_train, Y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(Y_test, predictions)
print("Accuracy:", acc)