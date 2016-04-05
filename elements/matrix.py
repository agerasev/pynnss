#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.array import Array
from pynn.element import Element


class MatrixElement(Element):
	def __init__(self, isize, osize, **kwargs):
		Element.__init__(self, [isize], [osize], **kwargs)
		self.isize = isize
		self.osize = osize


class Matrix(MatrixElement):
	def __init__(self, isize, osize, **kwargs):
		MatrixElement.__init__(self, isize, osize, **kwargs)

	class _State(Element._State):
		def __init__(self, isize, osize):
			arr = Array(0.01*np.random.randn(osize, isize))
			Element._State.__init__(self, arr)

	def newState(self):
		return self._State(self.isize, self.osize)

	class _Memory(Element._Memory):
		def __init__(self, isize):
			Element._Memory.__init__(self)
			self.idata = Array(isize)

	def newMemory(self):
		return self._Memory(self.isize)

	def _transmit(self, ctx):
		array.copy(ctx.mem.idata, ctx.src)
		array.dot(ctx.dst, ctx.state.data, ctx.src)

	def _backprop(self, ctx):
		if ctx.grad is not None:
			array.raddouter(ctx.grad.data, ctx.dst, ctx.mem.idata)
		array.dot(ctx.src, ctx.dst, ctx.state.data)
