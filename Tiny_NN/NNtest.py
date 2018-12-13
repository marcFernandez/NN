import NeuralNetwork
import NeuralNetwork4Input
import numpy as np


X4 = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y4 = np.array([[0],[1],[1],[0]])

X = np.array([[0,0,0],
              [0,1,0],
              [1,0,0],
              [1,1,0],
              [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0],[0],[1],[1],[0]])

NN = NeuralNetwork.NeuralNetwork(X,y)
NN4 = NeuralNetwork4Input.NeuralNetwork(X4,y4)

NN.predict(X)
print ""
NN4.predict(X4)
print ""

for i in range(500):
	NN.feedforward()
	NN.backprop()
	NN4.feedforward()
	NN4.backprop()

NN.predict(X)
print ""
NN4.predict(X4)
print ""

for i in range(500):
	NN.feedforward()
	NN.backprop()
	NN4.feedforward()
	NN4.backprop()

NN.predict(X)
print ""
NN4.predict(X4)
print ""

for i in range(500):
	NN.feedforward()
	NN.backprop()
	NN4.feedforward()
	NN4.backprop()

NN.predict(X)
print ""
NN4.predict(X4)
print ""

for i in range(500):
	NN.feedforward()
	NN.backprop()
	NN4.feedforward()
	NN4.backprop()

NN.predict(X)
print ""
NN4.predict(X4)
print ""

for i in range(10000):
	NN.feedforward()
	NN.backprop()
	NN4.feedforward()
	NN4.backprop()

NN.predict(X)
print ""
NN4.predict(X4)
print ""