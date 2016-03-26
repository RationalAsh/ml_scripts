#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist

class fuzzykmeans(object):
    def __init__(self, train_data, K, debug=True):
        '''train_data is the the dataset where each row is a sample
        and K is the number of clusters that need to be found.'''
        self.nclasses = K
        #Initialize K random means
        self.means = np.max(train_data)*np.random.randn(K, train_data.shape[1])
        #randinds = [i for i in range(train_data.shape[0])]
        #rinds = randinds[:K]
        #self.means = train_data[rinds,:]
        self.T = np.zeros((train_data.shape[0], K))
        self.DEBUG = debug
        self.costs = []

    def costf(self, train_data):
        '''Calculates the cost function on the current
        data set'''
        membs = self.m(train_data)
        dists = (cdist(self.means, train_data))**2
        return np.sum(membs*dists)

    def m(self, train_data, l=2):
        '''Calculates the membership function between every data
        point and every class. The parameter l determines the 
        fuzziness of the algorithm'''
        mems = cdist(self.means, train_data)
        mems = np.exp(-mems**2)
        return mems**l

    def train(self, train_data, epochs=10):
        '''Train the k-means algorithm until convergence.'''
        print("Epochs: ", epochs)
        self.costs = np.zeros(epochs)
        for j in range(epochs):

            print("Training Epoch %d:"%(j))

            #Calculate the cost function for plotting
            if self.DEBUG:
                J = self.costf(train_data)
                self.costs[j] = J
                print("Cost: "+str(J))
                print("Means: "+str(self.means))

            #Find the degree of membership between each points and 
            #each class
            membs = self.m(train_data)
            
            #Find the new centroids of the classes
            A = np.dot(membs, train_data)
            for i in range(self.nclasses):
                self.means[i,:] = A[i,:]/np.sum(membs[i,:])

if __name__=='__main__':
    #Generate the class Data
    CL1 = np.random.multivariate_normal([5, 0], 0.2*np.identity(2), 200)
    CL2 = np.random.multivariate_normal([-5,0], 0.2*np.identity(2), 200)
    CL3 = np.random.multivariate_normal([0, 5], 0.2*np.identity(2), 200)
    CL4 = np.random.multivariate_normal([0, 0], 0.2*np.identity(2), 200)
    CL5 = np.random.multivariate_normal([0, -5], 0.2*np.identity(2), 200)

    X = np.vstack((CL1, CL2, CL3))
    np.random.shuffle(X)

    #And plotting
    plt.scatter(X[:,0], X[:,1])
    plt.title('Unclassified data plot')
    plt.hold(True)

    km = fuzzykmeans(X, 3)
    km.train(X)
    plt.scatter(km.means[:,0], km.means[:,1], c='r', marker='s')
    plt.show()
