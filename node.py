#!/usr/bin/python3

# Node is structural unit of network
class Node:
	class Site:
		def __init__(self, size):
			self.size = size

	def __init__(self, nins, nouts):
		self.ins = [None]*nins
		self.outs = [None]*nouts

	def _gnins(self):
		return len(self.ins)
	nins = property(_gnins)

	def _gnouts(self):
		return len(self.outs)
	nouts = property(_gnouts)

	# stores state of node 
	class _Memory:
		vins = None
		vouts = None
		def __init__(self, vins, vouts):
			self.vins = vins
			self.vouts = vouts

	def Memory(self):
		return Node._Memory([None]*self.nins, [None]*self.nouts)

	# takes state and input data
	# returns new state and output data
	def _step(self, mem, vins):
		raise NotImplementedError()

	def step(self, mem, vins):
		(newmem, vouts) = self._step(mem, vins)
		newmem.vins = vins
		newmem.vouts = vouts
		return (newmem, vouts)

	# stores learn results
	class _Experience:
		eins = None
		eouts = None
		def __init__(self, nins, nouts):
			self.eins = [None]*nins
			self.eouts = [None]*nouts

	def Experience(self):
		return Node._Experience(self.nins, self.nouts)

	# takes state, experience and output error
	# modifies existing experience
	# returns input error

	def _backprop(self, exp, mem, eouts):
		raise NotImplementedError()

	def backprop(self, exp, mem, eouts):
		eins = self._backprop(exp, mem, eouts)
		exp.eins = eins
		exp.eouts = eouts
		return eins

	def learn(self, exp, rate):
		raise NotImplementedError()