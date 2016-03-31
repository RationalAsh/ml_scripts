#!/usr/bin/python

import numpy as np
from numpy import random, dot

#Sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)

def colwise_softmax(A):
    exps = np.exp(A)
    return exps/np.sum(exps,axis=0)

class ANN(object):
    def __init__(self, sizes):
        '''Initialize the neural network'''
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.hiddens = sizes[1:-1]
        self.wsizes = sizes[1:]
        self.weights = [random.randn(sz[0], sz[1]) for sz in zip(sizes[1:], sizes[:-1])]
        self.biases = [random.randn(sz,1) for sz in sizes[1:]]
        self.activations = []

    def feedforward(self, train_data):
        '''Perform the feedforward operation on the neural network'''
        self.activations = []
        A = train_data.T
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            #If not final layer, apply logistic sigmoid
            if i < len(self.weights)-1:
                A = sigmoid(np.dot(W,A) + b)
            else:
                A = colwise_softmax(np.dot(W,A) + b)
            self.activations.append(A)
        return A

    def backprop(train_data, train_targets):
        '''Uses the backpropogation algorithm to calculate the 
        gradients for the weights of each layer of the neural network.
        '''
        #First get the output of the final layer
        Y = self.feedforward(train_data).T
        #Calculate delta
        d = Y - train_targets

    def GD(self, train_data, train_targets, epochs=30, eta=0.01, debug=True):
        '''Trains the neural network using gradient descent. Uses backpropogation
        to compute the gradients of the weights and biases of the network. The 
        train_data should be in a matrix where each row is a data sample and the
        train_targets should be in the form of a matrix where each row is a one-of-K
        encoded binary vector that represents the class of the corresponding sample in
        train_data'''
        pass

    def SGD(self, traindata, testdata, epochs=30, batch_size=10, eta=3.0, debug=True):
        '''Training data and testing data is given as a tuple of inputs and targets
        of the form (X, T).'''
        #Learning rate
        eta = 0.1
        #Loop for number of epochs
        for i in range(epochs):
            if debug: print("Starting epoch 1...")
            #Generate mini batches
            random.shuffle(traindata)
            mini_batches = [traindata[k:k+batch_size] for k in range(0,len(traindata),batch_size)]
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, eta)
            if debug:
                print("Epoch 1 complete")
    

if __name__ == '__main__':
    CLASSES = 10
    TRAINING = 15000
    TESTING = 5000
    EPOCHS = 50

    INPUTS = 784
    HIDDEN0 = 30
    OUTPUTS = CLASSES
    EPOCHS = 40

    print("Loading the dataset...")
    DATA = np.genfromtxt('mnist_train.csv', delimiter=',', max_rows=200)
    print("Done")

    #Get the training data
    t = DATA[:TRAINING,0]
    X = DATA[:TRAINING,1:]/255.
    T = np.zeros((len(t), CLASSES))

    for i in range(t.shape[0]):
        T[i,int(t[i])] = 1

    #Get the testing data
    t1 = DATA[TRAINING:,0]
    X1 = DATA[TRAINING:,1:]/255.
    T1 = np.zeros((len(t1), CLASSES))

    for i in range(t1.shape[0]):
        T1[i,int(t1[i])] = 1

    #Initialize the neural network
    nn = ANN([INPUTS, HIDDEN0, OUTPUTS])
    
    #Training the neural network using Gradient descent
    for i in range(EPOCHS):
        #DO the feedforward thing.
        pass
