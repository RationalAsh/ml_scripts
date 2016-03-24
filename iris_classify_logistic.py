#!/usr/bin/python

#Script to classify Iris flowers using logistic regression

import numpy as np
import multiclass_logistic_reg as mclf
import time

NCLASSES = 3
CLASSES = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
EPOCHS = 50

#Load the data
DATA = np.loadtxt('Iris.csv', delimiter=',', dtype=str, skiprows=1)
#Separate the inputs
X = np.array(DATA[:,:5], dtype=float)
C = DATA[:,5]
T = np.zeros((X.shape[0], NCLASSES))

for i in range(X.shape[0]):
    idx = CLASSES[C[i]]
    T[i,idx] = 1

#Initialize the classifier
clf = mclf.classifier(X.shape[1], NCLASSES)

#Train using SGD for 50 epochs
for i in range(EPOCHS):
    clf.SGD(X, T, batch_size=10, epochs=1, eta=0.0001)
    clf.evalData(X, T)
    time.sleep(1)
