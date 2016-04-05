#!/usr/bin/python3

from pynn.element import Uniform, Tanh, Softmax

class Loss:
	def __init__(self, size):
		self.size = size
		self.target = None

class EuclideanLoss(Loss):
	def __init__(self, size):
		Loss.__init__(self, size)