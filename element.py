#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.array import Array
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
			def __init__(self, data):
				Node._State._Gradient.__init__(self)
				self.data = data

			def mul(self, factor):
				array.rmul(self.data, factor)

			def clip(self, value):
				array.rclip(self.data, -value, value)

		def newGradient(self):
			if self.data is not None:
				return self._Gradient(Array(np.zeros(self.data.shape)))
			return None

		class _RateConst(Node._State._Rate):
			def __init__(self, factor):
				Node._State._Rate.__init__(self)
				self.factor = factor

			def apply(self, dst, src):
				array.rsubmul(dst, src, self.factor)

		class _RateAdaGrad(_RateConst):
			def __init__(self, factor, data):
				Element._State._RateConst.__init__(self, factor)
				self.data = data

			def update(self, src):
				array.radd_adagrad(self.data, src)

			def apply(self, dst, src):
				array.rsub_adagrad(dst, src, self.factor, self.data)

		def newRate(self, factor, **kwargs):
			if kwargs.get('adagrad', False):
				return self._RateAdaGrad(factor, Array(np.zeros(self.data.shape) + 1e-6))
			else:
				return self._RateConst(factor)

		def learn(self, grad, rate):
			if grad is not None:
				rate.apply(self.data, grad.data)

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
