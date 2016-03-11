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
		def __init__(self):
			self.vins = None
			self.vouts = None

	def Memory(self):
		return Node._Memory()

	# takes state and input data
	# returns new state and output data
	def _feedforward(self, mem, vins):
		raise NotImplementedError()

	def feedforward(self, mem, vins):
		(nmem, vouts) = self._feedforward(mem, vins)
		nmem.vins = vins
		nmem.vouts = vouts
		return (nmem, vouts)

	# stores learn results
	class _Experience:
		def __init__(self, par):
			self.eins = None
			self.eouts = None
			self.count = 0

		def _clean(self):
			pass

		def clean(self):
			self.count = 0
			self._clean()

	def Experience(self, par):
		return Node._Experience(par)

	# takes state, experience and output error
	# modifies existing experience
	# returns input error

	def _backprop(self, exp, mem, eouts):
		raise NotImplementedError()

	def backprop(self, exp, mem, eouts):
		eins = self._backprop(exp, mem, eouts)
		exp.eins = eins
		exp.eouts = eouts
		exp.count += 1
		return eins

	def learn(self, exp, rate):
		raise NotImplementedError()