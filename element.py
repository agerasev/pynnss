#!/usr/bin/python3

import numpy as np
import pynn.array as array
from array import Array
from pynn.node import Node


# Element is a basic node
class Element(Node):
	class _State(Node._State):
		def __init__(self, data=None):
			Node._State.__init__(self)
			self.data = data

		def copyto(self, out):
			array.copy(out.data, self.data)

		class _Gradient(Node._State._Gradient):
			def __init__(self, data=None):
				Node._Gradient.__init__(self)
				self.data = data

			def mul(self, factor):
				array.rmul(self.data, factor)

			def clip(self, value):
				array.rclip(self.data, -value, value)

		def newGradient(self):
			if self.data is not None:
				return self._Gradient(Array(np.zeros_like(self.data)))
			return None

		def learn(self, grad, rate):
			if grad is not None:
				array.rsubmul(self.data, grad.data, rate.data)

	def newState(self):
		return None

	def __init__(self, isizes, osizes, **kwargs):
		isites = [self.Site(isize) for isize in isizes]
		osites = [self.Site(osize) for osize in osizes]
		Node.__init__(self, isites, osites, **kwargs)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def _backprop(self, ctx):
		raise NotImplementedError()
