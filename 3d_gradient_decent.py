#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np 


def function_z(x,y):
	return np.sin(5*x)*np.cos(5*y)/5
def dy_dx(x,y):
	return np.cos(5*x)*np.cos(5*x),-np.sin(5*x)*np.sin(5*y)
x = np.arange(-1,1,0.05)
y =np.arange(-1,1,0.05)
X,Y =np.meshgrid(x,y)
Z =function_z(X,Y)
ball = (0.7,0.4,function_z(0.7,0.4))
learning_rate =0.01
ax =plt.subplot(projection="3d",computed_zorder=False)
for _ in range(1000):
	x_dev ,y_dev = dy_dx(ball[0],ball[1])
	x_new,y_new=ball[0]-learning_rate*x_dev,ball[1]-learning_rate*y_dev
	ball = (x_new,y_new,function_z(x_new,y_new))
	print(ball[0],ball[1],ball[2])
	ax.plot_surface(X,Y,Z,cmap="viridis",zorder=0)
	ax.scatter(ball[0],ball[1],ball[2],color="magenta",zorder=1)
	plt.pause(0.001)
	ax.clear()
