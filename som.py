#!/usr/bin/python

#My implementation of self organizing maps

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import time


class som(object):
    def __init__(self, train_data, M=2, size=10):
        '''Initialize the Self Organizing Map
        train_data: Dataset where each row is an N dimensional
                    data point.
        M: The dimension of the lattice.
        size: The size of the lattice for the self organizing map.
        '''
        self.size = size
        self.dim = train_data.shape[1]
        #Randomly initialize the weights
        self.weights = np.random.uniform(0.0, 1.0, (size,size,train_data.shape[1]))

    def findBMU(self, x):
        '''Returns indices of the best matching unit'''
        dists = cdist(x, np.reshape(self.weights, (self.size**2, self.dim)))
        midx = np.argmin(dists)
        row = int(midx/self.size)
        col = midx%self.size

        return row, col

    def neighbourhood_func(self, row, col, spread=100):
        '''Returns an array of same size as the lattice with 
        neighbourhood weights
        The spread should be initially high but drop of slowly over time
        A good starting value is the size of the array'''
        x = np.arange(0,self.size,1)
        y = np.arange(0,self.size,1)
        xx,yy = np.meshgrid(x, y)
        z = np.exp(-((xx-col)**2+(yy-row)**2)/(2*spread**2))
        return z        

    def train(self, train_data, eta=0.1, iters=100):
        spread = self.size
        for j in range(iters):
            print("Epoch %d: "%(j))
            for i in range(train_data.shape[0]):
                x = train_data[i,:]
                #Find the best matching unit
                #print(x)
                row,col = self.findBMU(np.reshape(x, (1,self.dim)))
                #Adjust weights with neighbourhood function weighting
                ws = np.reshape(self.weights, (self.size**2, self.dim))
                ngd = np.reshape(self.neighbourhood_func(row, col, spread), (self.size**2, 1))
                wts = ngd*np.tile(eta, (self.size**2, 1))
                diffs = np.tile(x, (self.size**2, 1)) - ws
                ws = ws + wts*diffs
                self.weights = np.reshape(ws, (self.size, self.size, self.dim))
                #Adjust the neighbourhood size
                spread = spread*0.99997

if __name__=='__main__':
    #Generate a random dataset
    rands = np.random.uniform(0.5, 1.0, 2500)
    rands2 = np.random.uniform(0., 0.3, 2500)
    CL1 = 0.1*np.ones((2500,3))
    CL2 = 0.1*np.ones((2500,3))
    CL3 = 0.1*np.ones((2500,3))
    CL4 = 0.1*np.ones((2500,3))
    CL1[:,0] = rands
    CL2[:,1] = rands
    CL3[:,2] = rands
    CL4[:,2] = rands2
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.reshape(CL1, (50,50,3)), interpolation='none')
    plt.subplot(222)
    plt.imshow(np.reshape(CL2, (50,50,3)), interpolation='none')
    plt.subplot(223)
    plt.imshow(np.reshape(CL3, (50,50,3)), interpolation='none')
    plt.subplot(224)
    plt.imshow(np.reshape(CL4, (50,50,3)), interpolation='none')
    plt.show()
    
    X = np.vstack((CL1, CL2, CL3, CL4))
    np.random.shuffle(X)
    #X = np.random.uniform(0., 1., (10000,3))

    sm = som(X, 2, 50)
    sm.train(X, iters=10)
    plt.figure()
    plt.imshow(sm.weights, interpolation='none')
    plt.show()
    
    # spread = 50
    # eta = 0.1
    # fig = plt.figure()
    # plt.hold(True)
    # for i in range(100):
    #     print("Epoch %d"%(i))
    #     for i in range(X.shape[0]):
    #         x = X[i,:]
    #         row, col = sm.findBMU(np.reshape(x, (1, sm.dim)))
    #         #print(str(row)+','+str(col))
    #         ws = np.reshape(sm.weights, (sm.size**2, sm.dim))
    #         ngd = np.reshape(sm.neighbourhood_func(row, col, spread), (sm.size**2, 1))
    #         wts = ngd*np.tile(eta, (sm.size**2, 1))
    #         diffs = np.tile(x, (sm.size**2, 1)) - ws
    #         ws = ws + wts*diffs
    #         sm.weights = np.reshape(ws, (sm.size, sm.size, sm.dim))
    #         spread = spread*0.99997
    #         #eta = eta*0.99997
    #     time.sleep(0.5)
    #     plt.subplot(121)
    #     plt.imshow(sm.weights, interpolation='none')
    #     plt.subplot(122)
    #     plt.imshow(np.reshape(ngd, (sm.size, sm.size)), interpolation='none')
    #     plt.show()
