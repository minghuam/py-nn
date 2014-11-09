import numpy as np

class sigmoid:
	@staticmethod
	def f(z):
		return 1.0/(1.0 + np.exp(-z))

	@staticmethod
	def df(z):
		f = sigmoid.f(z) 
		return f * (1.0 - f)

class tanh:
	@staticmethod
	def f(z):
		return np.tanh(z)

	@staticmethod
	def df(z):
		f = np.tanh(z)
		return 1.0 - f*f