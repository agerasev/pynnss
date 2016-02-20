#!/usr/bin/python3

from node import Node

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

	def _vstep(self, vins):
		raise NotImplementedError()

	def _step(self, mem, vins):
		vouts = self._vstep(vins)
		return (Node._Memory(vins, vouts), vouts)

class Product(Element):
	weight = None
	bias = None

	def __init__(self, sin, sout):
		Element.__init__(self, [sin], [sout])
		self.randomize()

	def _gsin(self):
		return self.ins[0].size
	sin = property(_gsin)

	def _gsout(self):
		return self.outs[0].size
	sout = property(_gsout)

	def _vstep(self, vins):
		return [np.dot(vins[0], self.weight) + self.bias]

	class _Experience(Node._Experience):
		gweight = None
		gbias = None
		def __init__(self, sin, sout):
			Node._Experience.__init__(self, 1, 1)
			self.gweight = np.zeros((sin, sout))
			self.gbias = np.zeros((sout))

	def Experience(self):
		return Product._Experience(self.sin, self.sout)

	def _learn(self, exp, mem, eouts):
		exp.gbias += eouts[0]
		exp.gweight += np.outer(mem.vins[0], eouts[0])
		eins = [np.dot(self.weight, eouts[0])]
		return eins

	def randomize(self):
		self.weight = 2*np.random.rand(self.sin, self.sout) - 1
		self.bias = 2*np.random.rand(self.sout) - 1


class Scalar(Element):
	def __init__(self, size):
		Element.__init__(self, [size], [size])

	def _gsize(self):
		return len(self.ins[0].size)
	size = property(_gsize)


class Uniform(Scalar):
	def __init__(self, size):
		Scalar.__init__(self, size)
	
	def _vstep(self, vins):
		return [vins[0]]

	def _learn(self, exp, mem, eouts):
		return [eouts[0]]


class Sigmoid(Scalar):
	def __init__(self, size):
		Scalar.__init__(self, size)

	def _vstep(self, vins):
		return [np.tanh(vins[0])]

	def _learn(self, exp, mem, eouts):
		return [eouts[0]*(1/np.cosh(mem.vins[0]))**2]
