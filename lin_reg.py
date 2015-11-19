#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

#Mean Square error function
def costf(X, y, theta):
    m = y.shape[0]
    #print m
    return (1.0/m)*np.sum(np.power(np.dot(X,theta) - y, 2))

#Gradient of error function
def gradientf(X, y, theta):
    m = y.shape[0]
    err = np.dot(X, theta) - y
    return (2.0/m)*np.dot(np.transpose(X), err)


t = np.arange(0,10,0.01)
y = 2*square(t) + 0*np.random.random(t.shape)

X = np.array([[1, np.sin(x), np.sin(3*x), np.sin(5*x), np.sin(7*x), np.sin(9*x)] for x in t])
th = np.zeros(6)

errors = []
thetas = []

#Optimizing using gradient descent algorithm
numiters = 1000
alpha = 0.02    #Learning rate

errors.append(costf(X,y,th))

for i in xrange(numiters):
    #Gradient descent
    grad = gradientf(X,y,th)
    th = th - alpha*grad
    errors.append(costf(X,y,th))
    thetas.append(th)
    if(i%20 == 0):
        print "Iteration: "+str(i)
        print "Costf: "+ str(costf(X,y,th))
        print "Gradient: " + str(gradientf(X, t, th))
        print "Theta: "+ str(th)

y_ = np.dot(X, th)
#Closed form solution
th_opt = np.dot(np.linalg.pinv(X), y)
y_opt = np.dot(X, th_opt)

#Plotting results
plt.plot(t, y)
plt.ylim(-3,3)
plt.xlabel('x')
plt.ylabel('y')
plt.hold(True)
#plt.plot(t, y_)
#plt.plot(t, y_opt)

plt.figure()
plt.plot(errors)
plt.title("Error over time")
plt.ylabel("Error")
plt.xlabel("Number of iterations")
plt.show()
