#!/bin/python3
from Decision_tree import DecisionTree
from collections import Counter
import numpy as np
class RandomForest:
    def __init__(self,n_trees=10,max_depth=10,min_sample_split=2,n_features=None):
        self.n_trees=[]
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.n_features=n_features
    def fit(self,):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,min_samples_split=self.min_sample_split,n_feature=self.n_features)
            X_sample,Y_sample= self._bootstrap_samples(X,Y)
            tree.fit(X_sample,Y_sample)
            self.trees.append(tree)
    def _bootstrap_samples(self,X,Y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],Y[idxs]
    def _most_common_label(self,Y):
        counter = Counter(Y)
        most_common= counter.most_common(1)[0][0]
        return most_common

    def predict(self,X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions,0,1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions



#train.py
if __name__=="__main__":
	from sklearn import datasets 
	from sklearn.model_selection import train_test_split

	data = datasets.load_breast_cancer()
	X = data.data
	Y =data.target

	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
	def accuracy(Y_true,Y_pred):
    		accuracy = np.sum(Y_true==Y_pred)/len(Y_true)
    		return accuracy
	clf=RandomForest()
	clf.fit(X_train,Y_train)
	predictions = clf.predict(X_test)

	acc =accuracy(Y_test,predictions)
	print(acc)










