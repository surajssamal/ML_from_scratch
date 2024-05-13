#!/bin/python3
import numpy as np


class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.slope = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.slope) + self.intercept

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.slope = self.slope - self.lr * dw
            self.intercept = self.intercept - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.slope) + self.intercept
        return y_pred
