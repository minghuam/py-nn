"""
	Multilayer Perceptron
	http://en.wikipedia.org/wiki/Multilayer_perceptron
"""
import random
import numpy as np
import sys

# Multiple Layer Perceptron
class MLP:
	def __init__(self, sizes, afunc):
		"""
			'sizes' stands for layer counts and neuron sizes for each layer,
			e.g., [3,2,1] stands for a 3-layer network, with 3, 2, 1 neurons for each layer respectively
			note that input layer does not have bias		
		""" 
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.afunc = afunc;
		self.initialize_weights_uniform()
		#self.initialize_weights_gaussian(0.1)
		#self.initialize_weights_xavier()

	def initialize_weights_uniform(self):
		# uniform distribution between [-0.5, 0.5]
		self.weights = [np.random.uniform(-0.5, 0.5, [size2, size1]) for size1, size2 in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [np.zeros([size, 1]) for size in self.sizes[1:]]

	def initialize_weights_gaussian(self, std):
		# guassian distribution 
		self.weights = [np.random.normal(0, std, [size2, size1]) for size1, size2 in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [np.zeros([size, 1]) for size in self.sizes[1:]]

	def initialize_weights_xavier(self):
		"""
			Understanding the difficulty of training deep feedforward neural networks
			Xavier Glorot, Yoshua Bengio
			w ~ uniform(-1/sqrt(n), 1/sqrt(n)), where n is the size of the previous layer
		"""
		self.weights = [np.random.uniform(-1/sqrt(size1), 1/sqrt(size1)) for size1, size2 in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [np.zeros([size, ]) for size in self.sizes[1:]]
		
	def forward_propagation(self, x):
		"""
			y = wx+b
		"""
		for b, w in zip(self.biases, self.weights):
			x = self.afunc.f(np.dot(w, x) + b)
		return x

	def backward_propagation(self, x, y):
		"""
			calculate output and activation for each layer in the forward pass
			calculate error gradient_wrt_weight, gradient_wrt_bias in the backward pass
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# forward pass from input layer to output layer
		activations = [x]
		zs = []
		activation = x
		for b,w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self.afunc.f(z)
			activations.append(activation)

		# backward pass from output layer to input layer
		delta = (activations[-1] - y) * self.afunc.df(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# other layers
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			delta = np.dot(self.weights[-l+1].transpose(), delta) * self.afunc.df(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			batch = 0
			for mini_batch in mini_batches:
				batch = batch + 1
				self.update_mini_batch(mini_batch, eta, j+1, batch)
			if test_data is not None:
				print "Epoch {0} done: {1}/{2}".format(j, self.evaluate(test_data), len(test_data))
			else:
				print "Epoch {0} done.".format(j)

	def update_mini_batch(self, mini_batch, eta, epoch, batch):
		# backprop for each minibatch and sum up
		sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
		sum_nabla_w = [np.zeros(w.shape) for w in self.weights]
		data_index = 0
		for x,y in mini_batch:
			data_index = data_index + 1
			#print "epoch {0}, mini_batch {1}: {2}/{3}".format(epoch, batch, data_index, len(mini_batch))
			nabla_b, nabla_w = self.backward_propagation(x,y)
			sum_nabla_b = [b1 + b2 for b1,b2 in zip(nabla_b, sum_nabla_b)]
			sum_nabla_w = [w1 + w2 for w1,w2 in zip(nabla_w, sum_nabla_w)]

		# update weights and biases
		self.weights = [w-(eta/len(mini_batch))*sum_w for w,sum_w in zip(self.weights, sum_nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*sum_b for b,sum_b in zip(self.biases, sum_nabla_b)]

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.forward_propagation(x)), y) for (x,y) in test_data]
		return sum(int(x == y) for (x,y) in test_results)
