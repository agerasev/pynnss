#!/usr/bin/python3

from pynn.node import Node

import numpy as np

# Element is basic, zero-state node

class Element(Node):
	class _Gradient(Node._Gradient):
		def __init__(self, state):
			Node._Gradient.__init__(self)
			self.state = np.zeros_like(state)

		def mul(self, factor):
			self.state *= factor

		def clip(self, value):
			np.clip(self.state, -value, value, out=self.state)

	def newGradient(self):
		if self.state is None:
			return None
		return self._Gradient(self.state)

	def __init__(self, sins, souts, **kwargs):
		nins = len(sins)
		nouts = len(souts)
		Node.__init__(self, nins, nouts)
		for j in range(nins):
			self.ins[j] = Node.Site(sins[j])
		for j in range(nouts):
			self.outs[j] = Node.Site(souts[j])

		# initial state
		self.state = None
		if 'state' in kwargs.keys():
			self.state = kwargs['state']

		# complex numbers support
		self.complex = False
		if 'complex' in kwargs.keys():
			self.complex = kwargs['complex'] 

	def step(self, vins):
		raise NotImplementedError()

	def _transmit(self, state, vins):
		return self.step(vins)

	def backstep(self, grad, state, eouts):
		raise NotImplementedError()

	def _backprop(self, grad, error, state, eouts):
		return self.backstep(grad, state, eouts)

	def learn(self, grad, rate):
		if grad is not None:
			self.state -= rate.value*grad.state

class MatrixElement(Element):
	def __init__(self, sin, sout, **kwargs):
		Element.__init__(self, [sin], [sout], **kwargs)

	def _gsin(self):
		return self.ins[0].size
	sin = property(_gsin)

	def _gsout(self):
		return self.outs[0].size
	sout = property(_gsout)

# MatrixProduct multiplies input vector by matrix
class MatrixProduct(MatrixElement):
	def __init__(self, sin, sout, **kwargs):
		if 'state' not in kwargs.keys():
			if 'complex' in kwargs.keys() and kwargs['complex']:
				state = 0.01*(np.random.randn(sin, sout) + 1j*np.random.randn(sin, sout))
			else:
				state = 0.01*np.random.randn(sin, sout)
			kwargs['state'] = state

		MatrixElement.__init__(self, sin, sout, **kwargs)

	def _gweight(self):
		return self.state
	def _sweight(self, weight):
		self.state = weight
	weight = property(_gweight, _sweight)

	def step(self, vins):
		return [np.dot(vins[0], self.weight)]

	def backstep(self, grad, state, eouts):
		if grad is not None:
			grad.state += np.outer(state.vins[0], eouts[0])
		eins = [np.dot(self.weight, eouts[0])]
		return eins


class VectorElement(Element):
	def __init__(self, size, **kwargs):
		Element.__init__(self, [size], [size], **kwargs)

	def _gsize(self):
		return self.ins[0].size
	size = property(_gsize)


class Bias(VectorElement):
	def __init__(self, size, **kwargs):
		if 'state' not in kwargs.keys():
			if 'complex' in kwargs.keys() and kwargs['complex']:
				state = np.zeros(size, dtype=complex)
			else:
				state = np.zeros(size)
			kwargs['state'] = state

		VectorElement.__init__(self, size, **kwargs)

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
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)
	
	def step(self, vins):
		return [vins[0]]

	def backstep(self, grad, state, eouts):
		return [eouts[0]]


class Tanh(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def step(self, vins):
		return [np.tanh(vins[0])]

	def backstep(self, grad, state, eouts):
		return [eouts[0]*(1 - state.vouts[0]**2)]


class Softmax(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def step(self, vins):
		evin = np.exp(vins[0])
		return [evin/np.sum(evin)]

	def backstep(self, grad, state, eouts):
		raise NotImplementedError()

class Rectifier(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def step(self, vins):
		return [np.log(1 + np.exp(vins[0]))]

	def backstep(self, grad, state, eouts):
		e = np.exp(-state.vins[0])
		return [eouts[0]/(1 + e)]


class Mixer(Element):
	def __init__(self, size, nins, nouts, **kwargs):
		Element.__init__(self, [size]*nins, [size]*nouts, **kwargs)
		self.size = size

	def step(self, vins):
		if not self.complex:
			accum = np.zeros(self.size)
		else:
			accum = np.zeros(self.size, dtype=complex)

		for i in range(len(vins)):
			accum += vins[i]

		return [accum]*self.nouts

	def backstep(self, grad, state, eouts):
		if not self.complex:
			accum = np.zeros(self.size)
		else:
			accum = np.zeros(self.size, dtype=complex)

		for i in range(len(eouts)):
			accum += eouts[i]

		return [accum]*self.nins


class Fork(Mixer):
	def __init__(self, size, nouts, **kwargs):
		Mixer.__init__(self, size, 1, nouts, **kwargs)


class Join(Mixer):
	def __init__(self, size, nins, **kwargs):
		Mixer.__init__(self, size, nins, 1, **kwargs)
