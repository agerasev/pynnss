#!/usr/bin/python3

from time import clock


class Site:
	def __init__(self, size):
		self.size = size

	def __eq__(self, other):
		return self.size == other.size

	def __ne__(self, other):
		return not (self == other)


class Node:
	class Profiler:
		def __init__(self):
			self.start = 0
			self.time = 0

		def __enter__(self):
			self.start = clock()

		def __exit__(self, *args):
			self.time += clock() - self.start

	class _State:
		def __init__(self):
			pass

		def copyto(self, out):
			raise NotImplementedError()

		class _Memory:
			def __init__(self):
				pass

			def copyto(self, out):
				pass

		def newMemory(self, factory):
			return None

		class _Error:
			def __init__(self):
				pass

			def copyto(self, out):
				pass

		def newError(self, factory):
			return None

		class _Gradient:
			def __init__(self):
				pass

			def mul(self, factor):
				raise NotImplementedError()

			def clip(self, value):
				raise NotImplementedError()

			def clear(self):
				raise NotImplementedError()

		def newGradient(self, factory):
			return None

		class _Rate:
			def __init__(self):
				pass

		def newRate(self, factory):
			return self._Rate()

		def learn(self, grad, rate):
			raise NotImplementedError()

	def newState(self, factory):
		return None

	class _Trace:
		def __init__(self):
			pass

		def copyto(self, out):
			pass

	def newTrace(self, factory):
		return None

	class _Context:
		def __init__(self, src, dst, **kwargs):
			self.src = src
			self.dst = dst

			self.state = kwargs.get('state')
			self.trace = kwargs.get('trace')

			self.mem = kwargs.get('mem')
			self.err = kwargs.get('err')

			self.grad = kwargs.get('grad')
			self.rate = kwargs.get('rate')

	def newContext(self, factory, *args, **kwargs):
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
		if ctx.src is None or ctx.dst is None:
			raise Exception('src or dst is None')
		if self.fstat is not None:
			with self.fstat:
				self._transmit(ctx)
		else:
			self._transmit(ctx)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def backprop(self, ctx):
		if ctx.src is None or ctx.dst is None:
			raise Exception('src or dst is None')
		if self.bstat is not None:
			with self.bstat:
				self._backprop(ctx)
		else:
			self._backprop(ctx)

	def _backprop(self, ctx):
		raise NotImplementedError()
