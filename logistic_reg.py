#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

def sigmoid(a):
    return 1/(1+np.exp(-a))

#Total samples
N = 200
ITERS = 100
eta = 0.01

#Create random sample of 2D vectors for classes
C1 = np.random.multivariate_normal([5, 0], [[1, 0],[0, 1]], int(N/2))
C2 = np.random.multivariate_normal([0, 5], [[1, 0],[0, 1]], int(N/2))
t = np.hstack((np.ones((1,N/2)), np.zeros((1,N/2)))).T
X = np.hstack((np.vstack((C1, C2)), t))
np.random.shuffle(X)
t = X[:,2]
X = X[:,:2]
#X = np.hstack((np.ones((X.shape[0],1)),X))
w = np.random.multivariate_normal(np.zeros(X.shape[1]), np.identity(X.shape[1]))
b = 0.01

ERRS = np.zeros(ITERS)
ctr=0


#Plot the random data
plt.scatter(X[:,0], X[:,1], marker='x')
plt.title('Scatter plot of random data')
l = 10

#Find the optimal value of W using gradient descent
for i in range(ITERS):
    Y = sigmoid(np.dot(X,w) + b)
    error = -np.dot(t, np.nan_to_num(np.log(Y))) - np.dot((1-t), np.nan_to_num(np.log(1-Y))) + 0.5*l*(norm(w)**2)
    ERRS[i] = error
    grad = np.dot(X.T, Y - t) + l*w
    w = w - eta*grad
    b = b - eta*sum(Y-t)
    ctr += 1
    print(grad)
    print(error)
    print(w, b)
    #time.sleep(1)
    #print(ctr)
    #print(error)

#Plot the error over time
plt.figure()
plt.title('Error over time')
plt.plot(ERRS)
plt.xlabel('Iterations')
plt.ylabel('Error')

#Plot the decision boundary
t_pred = np.dot(X,w) + b
x = np.linspace(0,10,50)
y = -b/w[1] - (w[0]/w[1])*x
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='x')
plt.title('Scatter plot of random data')
plt.hold(True)
plt.plot(x, y)

plt.show()
