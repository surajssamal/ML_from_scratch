import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_regression_2 import LinearRegression

X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

fig = plt.figure(figsize=(8,6))
plt.scatter(X ,Y)
plt.show()

reg = LinearRegression(lr=0.01)
reg.fit(X,Y)



Y_pred_line = reg.predict(X)
plt.plot(X, Y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
