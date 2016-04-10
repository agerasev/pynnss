#!/usr/bin/python3

from pynn import array
from pynn.element.vector import Softmax


class Loss:
	class _Context:
		def __init__(self):
			self.target = None
			self.loss = 0.

	def __init__(self):
		pass


class SoftmaxLoss(Loss, Softmax):
	class _Context(Loss._Context, Softmax._Context):
		def __init__(self, node):
			Loss._Context.__init__(self)
			Softmax._Context.__init__(self, node)
			self.target = -1

	def newContext(self, factory):
		return self._Context(self)

	class _Trace(Softmax._Trace):
		def __init__(self, odata):
			Softmax._Trace.__init__(self)
			self.odata = odata

		def copyto(self, out):
			array.copy(out.odata, self.odata)

	def newTrace(self, factory):
		return self._Trace(factory.empty(self.size))

	def __init__(self, size, **kwargs):
		Loss.__init__(self)
		Softmax.__init__(self, size, **kwargs)

	def _transmit(self, ctx):
		Softmax._transmit(self, ctx)
		array.copy(ctx.trace.odata, ctx.dst)

	def _backprop(self, ctx):
		ctx.loss = array.softmaxloss(ctx.src, ctx.trace.odata, ctx.target)
