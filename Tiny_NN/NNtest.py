import NeuralNetwork
import numpy as np

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])

NN = NeuralNetwork.NeuralNetwork(X,y)
"""
for i in range(1500):
	NN.feedforward()
	NN.backprop()

print "Results after 1500 tests:"
print NN.output
"""
print "Prediction after 0 rounds of training"
NN.predict(X)

for i in range(500):
	NN.feedforward()
	NN.backprop()

print "Prediction after 500 rounds of training"
NN.predict(X)

for i in range(500):
	NN.feedforward()
	NN.backprop()

print "Prediction after 1000 rounds of training"
NN.predict(X)

for i in range(500):
	NN.feedforward()
	NN.backprop()

print "Prediction after 1500 rounds of training"
NN.predict(X)

for i in range(500):
	NN.feedforward()
	NN.backprop()

print "Prediction after 2000 rounds of training"
NN.predict(X)

for i in range(10000):
	NN.feedforward()
	NN.backprop()

print "Prediction after 12000 rounds of training"
NN.predict(X)