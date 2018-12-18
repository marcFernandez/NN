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
# NN4 = NeuralNetwork4Input.NeuralNetwork(X4,y4,True)
# Exemple de prediccio amb un entrenament insuficient
# Correr codi varis cops i veure com varia.

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

print "Prediction for input [0,1,0] in both nn's:"
NN.predict(np.array([0,1,0]))
print ""
NN4.predict(np.array([0,1,0]))
print "As we can see, the nn trained with the 4 inputs ended in 1 makes\na bad prediction."
print ""
print "Prediction for input [0,1,1] in both nn's:"
NN.predict(np.array([0,1,1]))
print ""
NN4.predict(np.array([0,1,1]))