#!/usr/bin/python3

import pynn.array as array
from pynn.array import Array
from pynn.element import Element


class Mixer(Element):
	class _Context(Element._Context):
		def __init__(self, size, *args, **kwargs):
			Element._Context.__init__(self, *args, **kwargs)
			self.accum = Array(size)

	def newContext(self, *args, **kwargs):
		return self._Context(self.size, *args, **kwargs)

	def __init__(self, size, inum, onum, **kwargs):
		Element.__init__(self, [size]*inum, [size]*onum, **kwargs)
		self.size = size

	def _transmit(self, ctx):
		array.copy(ctx.accum, ctx.src[0])
		for i in range(1, self.inum):
			array.radd(ctx.accum, ctx.src[i])

		for i in range(self.onum):
			array.copy(ctx.dst[i], ctx.accum)

	def _backprop(self, ctx):
		array.copy(ctx.accum, ctx.dst[0])
		for i in range(1, self.onum):
			array.radd(ctx.accum, ctx.dst[i])

		for i in range(self.inum):
			array.copy(ctx.src[i], ctx.accum)


class Fork(Mixer):
	def newContext(self, *args, **kwargs):
		return Element._Context(*args, **kwargs)

	def __init__(self, size, **kwargs):
		Mixer.__init__(self, size, 1, 2, **kwargs)

	def _transmit(self, ctx):
		array.copy(ctx.dst[0], ctx.src)
		array.copy(ctx.dst[1], ctx.src)

	def _backprop(self, ctx):
		array.add(ctx.src, ctx.dst[0], ctx.dst[1])


class Join(Mixer):
	def newContext(self, *args, **kwargs):
		return Element._Context(*args, **kwargs)

	def __init__(self, size, **kwargs):
		Mixer.__init__(self, size, 2, 1, **kwargs)

	def _transmit(self, ctx):
		array.add(ctx.dst, ctx.src[0], ctx.src[1])

	def _backprop(self, ctx):
		array.copy(ctx.src[0], ctx.dst)
		array.copy(ctx.src[1], ctx.dst)
