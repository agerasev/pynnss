#!/usr/bin/python3

import numpy as np
if __name__ == '__main__':
	import array
else:
	import pynn.array as array
from array import Array

if __name__ != '__main__':
	from pynn.node import Node
else:
	from node import Node


# Element is a basic node
class Element(Node):
	class _State(Node._State):
		def __init__(self, data=None):
			Node._State.__init__(self)
			self.data = data

		def copyto(self, out):
			array.copyto(out.data, self.data)

		class _Gradient(Node._State._Gradient):
			def __init__(self, data=None):
				Node._Gradient.__init__(self)
				self.data = data

			def mul(self, factor):
				self.data *= factor

			def clip(self, value):
				np.clip(self.data, -value, value, out=self.data)

		def newGradient(self):
			if self.data is not None:
				return self._Gradient(self, np.zeros_like(self.data))
			return None

		def learn(self, grad, rate):
			if grad is not None:
				self.data -= rate.data*grad.data

	def newState(self):
		return None

	def __init__(self, isite, osite, **kwargs):
		Node.__init__(self, isite, osite, **kwargs)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def _backprop(self, ctx):
		raise NotImplementedError()


class MatrixElement(Element):
	def __init__(self, sin, sout, **kwargs):
		Element.__init__(self, self.Site(sin), self.Site(sout), **kwargs)

	def _gsin(self):
		return self.isite.size
	sin = property(_gsin)

	def _gsout(self):
		return self.osite.size
	sout = property(_gsout)


# MatrixProduct multiplies input vector by matrix
class Matrix(MatrixElement):
	def __init__(self, sin, sout, **kwargs):
		MatrixElement.__init__(self, sin, sout, **kwargs)

	def newState(self):
		return self._State(0.01*np.random.randn(self.sin, self.sout))

	def _gweight(self):
		return self.data
	weight = property(_gweight)

	def _transmit(self, ctx):
		np.dot(ctx.din, self.weight, out=ctx.dout)

	def _backprop(self, ctx):
		if ctx.grad is not None:
			ctx.grad.data += np.outer(state.vins[0], eouts[0])
		eins = [np.dot(self.weight, eouts[0])]
		return eins


class VectorElement(Element):
	def __init__(self, size, state=None):
		Element.__init__(self, [size], [size], state)

	def _gsize(self):
		return self.ins[0].size
	size = property(_gsize)


class Bias(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size, np.zeros(size))

	def _gbias(self):
		return self.state
	def _sbias(self, bias):
		self.state = bias
	bias = property(_gbias, _sbias)

	def step(self, vins):
		return [vins[0] + self.bias]

	def backstep(self, grad, state, eouts):
		if grad is not None:
			grad.state += eouts[0]
		return [eouts[0]]


class Uniform(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)
	
	def step(self, vins):
		return [vins[0]]

	def backstep(self, grad, state, eouts):
		return [eouts[0]]


class Tanh(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)

	def step(self, vins):
		return [np.tanh(vins[0])]

	def backstep(self, grad, state, eouts):
		return [eouts[0]*(1 - state.vouts[0]**2)]


class Rectifier(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)

	def step(self, vins):
		return [np.log(1 + np.exp(vins[0]))]

	def backstep(self, grad, state, eouts):
		e = np.exp(-state.vins[0])
		return [eouts[0]/(1 + e)]


class Mixer(Element):
	def __init__(self, size, nins, nouts):
		Element.__init__(self, [size]*nins, [size]*nouts)
		self.size = size

	def step(self, vins):
		accum = np.zeros(self.size)
		for i in range(len(vins)):
			accum += vins[i]
		return [accum]*self.nouts

	def backstep(self, grad, state, eouts):
		accum = np.zeros(self.size)
		for i in range(len(eouts)):
			accum += eouts[i]
		return [accum]*self.nins


class Fork(Mixer):
	def __init__(self, size, nouts):
		Mixer.__init__(self, size, 1, nouts)


class Join(Mixer):
	def __init__(self, size, nins):
		Mixer.__init__(self, size, nins, 1)
