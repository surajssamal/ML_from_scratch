#!/bin/python3 
import numpy as np

np.random.seed(10)
class perceptron:
    def __init__(self,lr=0.01,no_iter=1000):
        self.lr=lr
        self.no_iter=no_iter
        self.weight =None
        self.bias =None
    def activation_function(self,x):
        return np.where(x>0,1,0)
    def fit(self,x,y):
        samples,feat = x.shape
        self.weight = np.random.randn(feat)
        self.bias = 0
        y = np.where(y>0,1,0)
        for _ in range(self.no_iter):
            for idx,x_i in enumerate(x):
                y_pred = np.dot(x_i,self.weight.T)+self.bias
                y_pred = self.activation_function(y_pred)

                #updating w and b

                self.weight +=self.lr*(y[idx]-y_pred)*x_i
                self.bias +=self.lr*(y[idx]-y_pred)

    def predict(self,x):
        y_pred = np.dot(x,self.weight.T)+self.bias
        y_pred =self.activation_function(y_pred)
        return y_pred

if __name__=="__main__":
    from nnfs.datasets import vertical
    from sklearn.model_selection import train_test_split
    x,y = vertical.create_data(50,2)
    import matplotlib.pyplot as plt
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.show()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=40)
    model = perceptron(lr=0.01,no_iter=1000)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    def accuracy(y_true,y_pred):
        accuracy = np.sum(y_true ==y_pred)/len(y_true)
        return accuracy
    print(accuracy(y_test,y_pred))
    



