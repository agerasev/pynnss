#!/usr/bin/python3

from time import clock


class NodeInfo:
	def __init__(self, isizes, osizes):
		if type(isizes) is int:
			isizes = [isizes]
		self.isizes = isizes
		self.inum = len(isizes)
		if type(osizes) is int:
			osizes = [osizes]
		self.osizes = osizes
		self.onum = len(osizes)


class Node(NodeInfo):
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
		def _gsrc(self):
			return self.srcs[0]

		def _ssrc(self, data):
			self.srcs[0] = data

		src = property(_gsrc, _ssrc)

		def _gdst(self):
			return self.dsts[0]

		def _sdst(self, data):
			self.dsts[0] = data

		dst = property(_gdst, _sdst)

		def __init__(self, node):
			self.node = node

			self.srcs = [None]*node.inum
			self.dsts = [None]*node.onum

			self.state = None
			self.trace = None

			self.mem = None
			self.err = None

			self.grad = None
			self.rate = None

	def newContext(self, factory):
		return self._Context(self)

	def __init__(self, isizes, osizes, **kwargs):
		NodeInfo.__init__(self, isizes, osizes)

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
