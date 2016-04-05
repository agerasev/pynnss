#!/usr/bin/python3

from time import clock


# Node is structural unit of network
class Node:
	# describes node sites
	class Site:
		def __init__(self, size, dtype=float):
			self.size = size
			self.dtype = dtype

	class Profiler:
		def __init__(self):
			self.start = 0
			self.time = 0

		def __enter__(self):
			self.start = clock()

		def __exit__(self, *args):
			self.time += clock() - self.start

	# stores node state like biases or matrix weights
	class _State:
		def __init__(self):
			pass

		def copyto(self, out):
			raise NotImplementedError()

		# stores learn results
		class _Gradient:
			def __init__(self):
				pass

			def mul(self, factor):
				raise NotImplementedError()

			def clip(self, value):
				raise NotImplementedError()

		def newGradient(self):
			return None

		def learn(self, grad, rate):
			raise NotImplementedError()

	def newState(self):
		return None

	# stores node memory e.g. inputs, outputs and data in loopbacks
	class _Memory:
		def __init__(self):
			pass

		def copyto(self, out):
			pass

	def newMemory(self):
		return None

	# stores node errors
	class _Error:
		def __init__(self):
			pass

		def copyto(self, out):
			pass

	def newError(self):
		return None

	# context for evaluation
	class _Context:
		def __init__(self, src, dst, **kwargs):
			self.src = src
			self.dst = dst
			self.state = kwargs.get('state')
			self.mem = kwargs.get('mem')

			if kwargs.get('learn', True):
				self.grad = kwargs.get('grad')
				self.err = kwargs.get('err')
				self.rate = kwargs.get('rate')

	def newContext(self, *args, **kwargs):
		return self._Context(*args, **kwargs)

	def __init__(self, isites, osites, **kwargs):
		self.isites = isites
		self.inum = len(isites)
		self.osites = osites
		self.onum = len(osites)

		if kwargs.get('prof', False):
			self.fstat = self.Profiler()
			self.bstat = self.Profiler()
		else:
			self.fstat = None
			self.bstat = None

	def transmit(self, ctx):
		if self.fstat is not None:
			with self.fstat:
				self._transmit(ctx)
		else:
			self._transmit(ctx)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def backprop(self, ctx):
		if self.bstat is not None:
			with self.bstat:
				self._backprop(ctx)
		else:
			self._backprop(ctx)

	def _backprop(self, ctx):
		raise NotImplementedError()
