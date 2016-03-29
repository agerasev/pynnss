#!/usr/bin/python3

from time import clock

# Node is structural unit of network
class Node:
	# describes node sites
	class Site:
		def __init__(self, size):
			self.size = size

	class Profiler:
		def __init__(self):
			self.start = 0
			self.time = 0

		def __enter__(self):
			self.start = clock()

		def __exit__(self, a,b,c):
			self.time += clock() - self.start

	# stores state of node 
	class _State:
		def __init__(self):
			self.vins = None
			self.vouts = None

		def __copy__(self):
			state = type(self)()
			state.vins = self.vins
			state.vouts = self.vouts
			return state

	def newState(self):
		return self._State()

	# stores node errors
	class _Error:
		def __init__(self):
			self.eins = None
			self.eouts = None

		def __copy__(self):
			error = type(self)()
			error.eins = self.eins
			error.eouts = self.eouts
			return error

	def newError(self):
		return self._Error()

	# stores learn results
	class _Gradient:
		def __init__(self):
			pass

		def mul(self, factor):
			raise NotImplementedError()
			
		def clip(self, value):
			raise NotImplementedError()

	def newGradient(self):
		return self._Gradient()


	def __init__(self, nins, nouts):
		self.ins = [None]*nins
		self.outs = [None]*nouts
		self.fprof = self.Profiler()
		self.bprof = self.Profiler()

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
		if len(vins) != self.nins:
			raise Exception('wrong inputs number')

		with self.fprof:
			vouts = self._transmit(state, vins)
		state.vins = vins
		state.vouts = vouts
		return vouts

	def _transmit(self, state, vins):
		raise NotImplementedError()

	# takes state, error state, gradient and output error
	# modifies existing gradient and error state
	# returns input error
	def backprop(self, grad, error, state, eouts):
		if len(eouts) != self.nouts:
			raise Exception('wrong inputs number')
		with self.bprof:
			eins = self._backprop(grad, error, state, eouts)
		error.eins = eins
		error.eouts = eouts
		return eins

	def _backprop(self, grad, error, state, eouts):
		raise NotImplementedError()

	def learn(self, grad, rate):
		raise NotImplementedError()