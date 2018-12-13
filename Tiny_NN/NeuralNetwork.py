import numpy as np
import random

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork():

	def __init__(self,x,y):
		self.input = x
		#self.weights1 = np.random.rand(self.input.shape[1],4)
		#self.weights2 = np.random.rand(4,1)
		self.weights1 = np.zeros((self.input.shape[1],self.input.shape[0]))
		for i in range(self.weights1.shape[0]):
			for j in range(self.weights1.shape[1]):
				self.weights1[i][j] = random.random()
		self.weights2 = np.zeros((4,1))
		for i in range(self.weights2.shape[0]):
			for j in range(self.weights2.shape[1]):
				self.weights2[i][j] = random.random()

		print self.weights1
		print self.weights2
		self.y = y
		self.output = np.zeros(y.shape)
		self.count = 0

	def feedforward(self,msg=False):
		# In this case, we are assuming that the layer biases are 0
		self.layer_1 = sigmoid(np.dot(self.input,self.weights1))
		self.output = sigmoid(np.dot(self.layer_1, self.weights2))
		self.count += 1

		# I've set up some code to see how many tests you've done and
		# the current error. (It'll only print them every 100 steps)
		if msg:
			if(self.count%100==0):
				print "Error at testing round "+str(self.count)
				print (self.y - self.output)**2

	def backprop(self):
		# We use the chain rule to find derivative of the loss function with respect
		# weights2 and weights1
		d_weights2 = np.dot(self.layer_1.T,(2*(self.y-self.output)*
							sigmoid_derivative(self.output)))

		d_weights1 = np.dot(self.input.T,(np.dot(2*(self.y-self.output)*
							sigmoid_derivative(self.output),self.weights2.T)*
							sigmoid_derivative(self.layer_1)))

		# Then we update the wheights
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	# Prints the predicted results from an input "input" with the current values
	# of self.weights1 and self.weights2
	def predict(self,input):
		self.layer_1 = sigmoid(np.dot(input,self.weights1))
		self.output = sigmoid(np.dot(self.layer_1, self.weights2))
		print self.output