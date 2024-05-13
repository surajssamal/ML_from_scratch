#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def function(x):
	return np.sin(x)

def function_derivative(x):
	return np.cos(x)
x = np.arange(-5,5,0.1)
y = function(x)

ball = (1,function(1))
learning_rate = 0.1

for _ in range(1000):
	new_x = ball[0]-learning_rate*function_derivative(ball[0])
	new_y = function(new_x)
	ball = (new_x,new_y)
	print(ball[0],ball[1],end="\n")
	plt.plot(x,y)
	plt.scatter(ball[0],ball[1],color="black")
	plt.pause(0.001)
	plt.clf()

