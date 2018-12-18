import numpy as np
import random

# We start implementing the sigmoid function and its derivative

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

# Then let's go with the main class

class NeuralNetwork():

	"""
	We need to give our neural network an input data 'x' and the expected value for it 'y'
	"""
	def __init__(self,x,y,copied=False):


		self.input = x
		self.y = y
		self.output = np.zeros(y.shape) # The output is initialized with 0's for future updates
		self.count = 0 # This counter will help us to check in which training episode we are

		if copied:
			"""
			We can initialize all the layer weights randomly but in purpose to learn a bit more
			about nn's i've implemented a way to start each neuron weight with some wanted value
			"""
			self.weights1 = np.random.rand(self.input.shape[1],self.y.shape[0])
			self.weights2 = np.random.rand(self.y.shape[0],1)

		else:
			self.weights1 = np.zeros((self.input.shape[1],self.input.shape[0]))
			for i in range(self.weights1.shape[0]):
				for j in range(self.weights1.shape[1]):
					self.weights1[i][j] = 2-i # random.random(); if u want it to be random [0,1]
			self.weights2 = np.zeros((self.y.shape[0],1))
			for i in range(self.weights2.shape[0]):
				for j in range(self.weights2.shape[1]):
					self.weights2[i][j] = i # random.random(); if u want it to be random [0,1]

		# If u want to check the initialized weights, just print them
		
		print self.weights1
		print self.weights2
		
		

	"""
	This feedforward function just gets the input and makes it go through all the layers until
	the output one.
	Then, if we choose it, will print every 'step' steps the current testing round and its loss
	"""
	def feedforward(self,msg=False,step=100):
		# In this case, we are assuming that the layer biases are 0
		self.layer_1 = sigmoid(np.dot(self.input,self.weights1))
		self.output = sigmoid(np.dot(self.layer_1, self.weights2))
		self.count += 1

		# I've set up some code to see how many tests you've done and
		# the current error. It'll only print them every 'step' steps
		# (it's set to 100 by default)
		if msg:
			if(self.count%100==0):
				print "Error at testing round "+str(self.count)
				print (self.y - self.output)**2

	"""
	The backprop method uses the chain rule to find derivative of the loss function with respect
	weights2 and weights1, so it can update all weights from the nn
	"""
	def backprop(self):
		# To do so, it uses 'sigmoid_derivative' function coded at the beginning
		d_weights2 = np.dot(self.layer_1.T,(2*(self.y-self.output)*
							sigmoid_derivative(self.output)))

		d_weights1 = np.dot(self.input.T,(np.dot(2*(self.y-self.output)*
							sigmoid_derivative(self.output),self.weights2.T)*
							sigmoid_derivative(self.layer_1)))

		# time to update
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	# Prints the predicted results from an input 'input' with the current values
	# of self.weights1 and self.weights2
	"""
	To check the prediction we just feedforward the input through the nn
	"""
	def predict(self,input):
		self.layer_1 = sigmoid(np.dot(input,self.weights1))
		self.output = sigmoid(np.dot(self.layer_1, self.weights2))
		print "After",self.count,"testing rounds, the prediction for 4Input nn is:"
		print self.output