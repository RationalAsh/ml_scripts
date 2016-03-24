import multiclass_logistic_reg as mclf
import numpy as np
import matplotlib.pyplot as plt

CLASSES = 10
TRAINING = 15000
TESTING = 5000
EPOCHS = 50

print("Loading the dataset...")
DATA = np.genfromtxt('mnist_train.csv', delimiter=',', max_rows=20000)
print("Done")

#Get the training data
t = DATA[:TRAINING,0]
X = DATA[:TRAINING,1:]/255.
T = np.zeros((len(t), CLASSES))

for i in range(t.shape[0]):
    T[i,t[i]] = 1

#Get the testing data
t1 = DATA[TRAINING:,0]
X1 = DATA[TRAINING:,1:]/255.
T1 = np.zeros((len(t1), CLASSES))

for i in range(t1.shape[0]):
    T1[i,t1[i]] = 1

#Initialize the classifier
clf = mclf.classifier(X.shape[1], CLASSES)

#Do the epochs manually so I can see what's going on
for i in range(EPOCHS):
    clf.SGD(X, T, epochs=1, eta=0.001, batch_size=1000)
    clf.evalData(X1, T1)

#Plot the weights to visualize them
# plt.imshow(np.reshape(clf.W[:,0], (28,28)))
# plt.imshow(np.reshape(clf.W[:,1], (28,28)))
# plt.imshow(np.reshape(clf.W[:,2], (28,28)))
# plt.imshow(np.reshape(clf.W[:,3], (28,28)))
# plt.imshow(np.reshape(clf.W[:,4], (28,28)))
# plt.imshow(np.reshape(clf.W[:,5], (28,28)))
# plt.imshow(np.reshape(clf.W[:,6], (28,28)))
# plt.imshow(np.reshape(clf.W[:,7], (28,28)))
# plt.imshow(np.reshape(clf.W[:,8], (28,28)))
# plt.imshow(np.reshape(clf.W[:,9], (28,28)))
