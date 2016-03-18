#!/usr/bin/python3

# Node is structural unit of network
class Node:
	# describes node sites
	class Site:
		def __init__(self, size):
			self.size = size

	# stores state of node 
	class State:
		def __init__(self):
			self.vins = None
			self.vouts = None

		def __copy__(self):
			return Node.State()

	def newState(self):
		return self.State()

	# stores node errors
	class Error:
		def __init__(self):
			self.eins = None
			self.eouts = None

		def __copy__(self):
			return Node.Error()

	def newError(self):
		return self.Error()

	# stores learn results
	class Gradient:
		def __init__(self):
			pass

		def mul(self, factor):
			raise NotImplementedError()
			
		def clip(self, value):
			raise NotImplementedError()

	def newGradient(self):
		return self.Gradient()


	def __init__(self, nins, nouts):
		self.ins = [None]*nins
		self.outs = [None]*nouts

	def _gnins(self):
		return len(self.ins)
	nins = property(_gnins)

	def _gnouts(self):
		return len(self.outs)
	nouts = property(_gnouts)


	# takes state and inputs
	# changes state
	# returns outputs
	def transmit(self, state, vins):
		vouts = self._transmit(state, vins)
		state.vins = vins
		state.vouts = vouts
		return vouts

	def _transmit(self, state, vins):
		raise NotImplementedError()

	# takes state, error state, gradient and output error
	# modifies existing gradient and error state
	# returns input error
	def backprop(self, state, error, grad, eouts):
		eins = self._backprop(exp, mem, eouts)
		exp.eins = eins
		exp.eouts = eouts
		exp.count += 1
		return eins

	def _backprop(self, grad, error, state, eouts):
		raise NotImplementedError()

	def learn(self, grad, rate):
		raise NotImplementedError()