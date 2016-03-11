#!/usr/bin/python3

from pynn.node import Node

import numpy as np

# Element is basic, zero-state node

class Element(Node):
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

	def _feedforward(self, mem, vins):
		vouts = self.step(vins)
		return (Node._Memory(), vouts)

	def backstep(self, exp, mem, eouts):
		raise NotImplementedError()

	def _backprop(self, exp, mem, eouts):
		return self.backstep(exp, mem, eouts)

# Matrix multiplies input vector by matrix
class Matrix(Element):
	def __init__(self, sin, sout):
		Element.__init__(self, [sin], [sout])
		self.weight = None
		self.randomize()

	def _gsin(self):
		return self.ins[0].size
	sin = property(_gsin)

	def _gsout(self):
		return self.outs[0].size
	sout = property(_gsout)

	def step(self, vins):
		return [np.dot(vins[0], self.weight)]

	class _Experience(Node._Experience):
		def __init__(self, par, sin, sout):
			Node._Experience.__init__(self, par)
			self.gweight = np.zeros((sin, sout))

			self.clip = par["clip"]
			self.adagrad = par["adagrad"]

			if self.adagrad:
				self.aweight = 1e-6 + np.zeros_like(self.gweight)

		def _clean(self):
			self.gweight = np.zeros_like(self.gweight)

	def Experience(self, par):
		return Matrix._Experience(par, self.sin, self.sout)

	def backstep(self, exp, mem, eouts):
		exp.gweight += np.outer(mem.vins[0], eouts[0])
		eins = [np.dot(self.weight, eouts[0])]
		return eins

	def learn(self, exp, rate):
		nweight = np.clip(exp.gweight/exp.count, -exp.clip, exp.clip)
		if exp.adagrad:
			exp.aweight += nweight**2
			rate = rate/np.sqrt(exp.aweight)
		self.weight -= rate*nweight

	def randomize(self):
		self.weight = 2*np.random.rand(self.sin, self.sout) - 1


class Vector(Element):
	def __init__(self, size):
		Element.__init__(self, [size], [size])

	def _gsize(self):
		return self.ins[0].size
	size = property(_gsize)


class Bias(Vector):
	def __init__(self, size):
		Vector.__init__(self, size)
		self.bias = None
		self.randomize()

	def step(self, vins):
		return [vins[0] + self.bias]

	class _Experience(Node._Experience):
		def __init__(self, par, size):
			Node._Experience.__init__(self, par)
			self.gbias = np.zeros(size)

			self.clip = par["clip"]
			self.adagrad = par["adagrad"]

			if self.adagrad:
				self.abias = 1e-6 + np.zeros_like(self.gbias)

		def _clean(self):
			self.gbias = np.zeros_like(self.gbias)

	def Experience(self, par):
		return Bias._Experience(par, self.size)

	def backstep(self, exp, mem, eouts):
		exp.gbias += eouts[0]
		return [eouts[0]]

	def learn(self, exp, rate):
		nbias = np.clip(exp.gbias/exp.count, -exp.clip, exp.clip)
		if exp.adagrad:
			exp.abias += nbias**2
			rate = rate/np.sqrt(exp.abias)
		self.bias -= rate*nbias

	def randomize(self):
		self.bias = 2*np.random.rand(self.size) - 1


class Uniform(Vector):
	def __init__(self, size):
		Vector.__init__(self, size)
	
	def step(self, vins):
		return [vins[0]]

	def backstep(self, exp, mem, eouts):
		return [eouts[0]]

	def learn(self, exp, rate):
		pass


class Tanh(Vector):
	def __init__(self, size):
		Vector.__init__(self, size)

	def step(self, vins):
		return [np.tanh(vins[0])]

	def backstep(self, exp, mem, eouts):
		return [eouts[0]*(1/np.cosh(mem.vins[0]))**2]

	def learn(self, exp, rate):
		pass

class Rectifier(Vector):
	def __init__(self, size):
		Vector.__init__(self, size)

	def step(self, vins):
		return [np.log(1 + np.exp(vins[0]))]

	def backstep(self, exp, mem, eouts):
		e = np.exp(-mem.vins[0])
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

	def backstep(self, exp, mem, eouts):
		accum = np.zeros(self.size)
		for i in range(len(eouts)):
			accum += eouts[i]
		return [accum]*self.nins

	def learn(self, exp, rate):
		pass


class Fork(Mixer):
	def __init__(self, size, nouts):
		Mixer.__init__(self, size, 1, nouts)


class Merger(Mixer):
	def __init__(self, size, nins):
		Mixer.__init__(self, size, nins, 1)
