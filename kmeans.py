#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.spatial.distance import pdist, cdist

#My implementation of the k-means clustering algorithm

class kmeans(object):
    def __init__(self, train_data, K, debug=True):
        '''train_data is the the dataset where each row is a sample
        and K is the number of clusters that need to be found.'''
        self.nclasses = K
        #Initialize K random means
        #self.means = np.random.randn(K, train_data.shape[1])
        randinds = [i for i in range(train_data.shape[0])]
        rinds = randinds[:K]
        self.means = train_data[rinds,:]
        self.T = np.zeros((train_data.shape[0], K))
        self.DEBUG = debug
        self.costs = []

    def costf(self, train_data):
        '''Calculates the cost function on the current
        data set'''
        dists = cdist(self.means, train_data)
        return sum([np.dot(a,b) for a,b in zip(dists.T, self.T)])

    def train(self, train_data, epochs=10):
        '''Train the k-means algorithm until convergence.'''
        print("Epochs: ", epochs)
        self.costs = np.zeros(epochs)
        for j in range(epochs):
            #Calculate the cost function for plotting
            if self.DEBUG:
                J = self.costf(train_data)
                self.costs[j] = J
                print("Cost: "+str(J))
                print("Means: "+str(self.means))

            print("Training Epoch %d:"%(j))
            #Assign each data point to the mean closest to it
            self.T = np.zeros((train_data.shape[0], self.nclasses))
            dists = cdist(self.means, train_data)
            cls = np.argmin(dists, axis=0)
            for i in range(len(cls)):
                self.T[i,cls[i]] = 1.

            #Find the new centroids of the classes
            for i in range(self.nclasses):
                cls_i = [n for n in range(train_data.shape[0]) if self.T[n,i]==1]
                self.means[i,:] = np.mean(train_data[cls_i, :], axis=0)

if __name__=='__main__':
    #Generate the class Data
    CL1 = np.random.multivariate_normal([5, 0], 0.2*np.identity(2), 200)
    CL2 = np.random.multivariate_normal([-5,0], 0.2*np.identity(2), 200)
    CL3 = np.random.multivariate_normal([0, 5], 0.2*np.identity(2), 200)
    CL4 = np.random.multivariate_normal([0, 0], 0.2*np.identity(2), 200)
    CL5 = np.random.multivariate_normal([0, -5], 0.2*np.identity(2), 200)

    X = np.vstack((CL1, CL2, CL3, CL4, CL5))
    np.random.shuffle(X)

    #And plotting
    plt.scatter(X[:,0], X[:,1])
    plt.title('Unclassified data plot')
    plt.hold(True)

    km = kmeans(X, 5)
    km.train(X)
    for m in km.means:
        plt.scatter(m[0], m[1], marker='s', c='r')
    plt.show()
