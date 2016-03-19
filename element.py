#!/usr/bin/python3

from pynn.node import Node

import numpy as np

# Element is basic, zero-state node

class Element(Node):
	class _Gradient(Node._Gradient):
		def __init__(self):
			Node._Gradient.__init__(self)

		def mul(self, factor):
			pass

		def clip(self, value):
			pass

	def __init__(self, sins, souts):
		nins = len(sins)
		nouts = len(souts)
		Node.__init__(self, nins, nouts)
		for j in range(nins):
			self.ins[j] = Node.Site(sins[j])
		for j in range(nouts):
			self.outs[j] = Node.Site(souts[j])

	def step(self, vins):
		raise NotImplementedError()

	def _transmit(self, state, vins):
		return self.step(vins)

	def backstep(self, grad, state, eouts):
		raise NotImplementedError()

	def _backprop(self, grad, error, state, eouts):
		return self.backstep(grad, state, eouts)

class MatrixElement(Element):
	class _Gradient(Element._Gradient):
		def __init__(self, sin, sout):
			self.sin = sin
			self.sout = sout

	def newGradient(self):
		return self._Gradient(self.sin, self.sout)

	def __init__(self, sin, sout):
		Element.__init__(self, [sin], [sout])

	def _gsin(self):
		return self.ins[0].size
	sin = property(_gsin)

	def _gsout(self):
		return self.outs[0].size
	sout = property(_gsout)

# MatrixProduct multiplies input vector by matrix
class MatrixProduct(MatrixElement):
	class _Gradient(MatrixElement._Gradient):
		def __init__(self, sin, sout):
			MatrixElement._Gradient.__init__(self, sin, sout)
			self.weight = np.zeros((sin, sout))

		def mul(self, factor):
			self.weight *= factor

		def clip(self, value):
			self.weight = np.clip(self.weight, -value, value)

	def __init__(self, sin, sout):
		MatrixElement.__init__(self, sin, sout)
		self.randomize()

	def step(self, vins):
		return [np.dot(vins[0], self.weight)]

	def backstep(self, grad, state, eouts):
		if grad is not None:
			grad.weight += np.outer(state.vins[0], eouts[0])
		eins = [np.dot(self.weight, eouts[0])]
		return eins

	def learn(self, grad, rate):
		#	exp.aweight += nweight**2
		#	rate = rate/np.sqrt(exp.aweight)
		self.weight -= rate*grad.weight

	def randomize(self):
		self.weight = 2*np.random.rand(self.sin, self.sout) - 1


class VectorElement(Element):
	class _Gradient(Element._Gradient):
		def __init__(self, size):
			Element._Gradient.__init__(self)
			self.size = size

	def newGradient(self):
		return self._Gradient(self.size)

	def __init__(self, size):
		Element.__init__(self, [size], [size])

	def _gsize(self):
		return self.ins[0].size
	size = property(_gsize)


class Bias(VectorElement):
	class _Gradient(VectorElement._Gradient):
		def __init__(self, size):
			VectorElement._Gradient.__init__(self, size)
			self.bias = np.zeros(size)

		def mul(self, factor):
			self.bias *= factor

		def clip(self, value):
			self.bias = np.clip(self.bias, -value, value)

	def __init__(self, size):
		VectorElement.__init__(self, size)
		self.randomize()

	def step(self, vins):
		return [vins[0] + self.bias]

	def backstep(self, grad, state, eouts):
		if grad is not None:
			grad.bias += eouts[0]
		return [eouts[0]]

	def learn(self, grad, rate):
		#	exp.abias += nbias**2
		#	rate = rate/np.sqrt(exp.abias)
		self.bias -= rate*grad.bias

	def randomize(self):
		self.bias = 2*np.random.rand(self.size) - 1


class Uniform(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)
	
	def step(self, vins):
		return [vins[0]]

	def backstep(self, grad, state, eouts):
		return [eouts[0]]

	def learn(self, grad, rate):
		pass


class Tanh(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)

	def step(self, vins):
		return [np.tanh(vins[0])]

	def backstep(self, grad, state, eouts):
		return [eouts[0]*(1/np.cosh(state.vins[0]))**2]

	def learn(self, grad, rate):
		pass

class Rectifier(VectorElement):
	def __init__(self, size):
		VectorElement.__init__(self, size)

	def step(self, vins):
		return [np.log(1 + np.exp(vins[0]))]

	def backstep(self, grad, state, eouts):
		e = np.exp(-state.vins[0])
		return [eouts[0]/(1 + e)]

	def learn(self, exp, rate):
		pass

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

	def learn(self, grad, rate):
		pass


class Fork(Mixer):
	def __init__(self, size, nouts):
		Mixer.__init__(self, size, 1, nouts)


class Merger(Mixer):
	def __init__(self, size, nins):
		Mixer.__init__(self, size, nins, 1)
