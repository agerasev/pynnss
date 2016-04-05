#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.array import Array
from pynn.element import Element


class VectorElement(Element):
	def __init__(self, size, **kwargs):
		Element.__init__(self, [size], [size], **kwargs)
		self.size = size


class Bias(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def newState(self):
		return self._State(Array(np.zeros(self.size)))

	def _transmit(self, ctx):
		array.add(ctx.dst, ctx.src, ctx.state.data)

	def _backprop(self, ctx):
		if ctx.grad is not None:
			array.radd(ctx.grad.data, ctx.dst)
		array.copy(ctx.src, ctx.dst)


class Uniform(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def _transmit(self, ctx):
		array.copy(ctx.dst, ctx.src)

	def _backprop(self, ctx):
		array.copy(ctx.src, ctx.dst)


class Tanh(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	class _Memory(Element._Memory):
		def __init__(self, osize):
			Element._Memory.__init__(self)
			self.odata = Array(osize)

	def newMemory(self):
		return self._Memory(self.odata)

	def _transmit(self, ctx):
		array.tanh(ctx.dst, ctx.src)

	def _backprop(self, ctx):
		array.bptanh(ctx.src, ctx.dst, ctx.mem.odata)

# TODO:
# class Softmax(VectorElement)
# class Rectifier(VectorElement)
