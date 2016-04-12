#!/usr/bin/python3

import numpy as np


class Batch:
	def __init__(self, factory, size, maxlen=1):
		self.size = size
		self.maxlen = maxlen
		self.factory = factory

	def do(self, net, ctx, data):
		factory = self.factory
		loss = 0
		ctx.grad.clear()
		src = factory.empty(net.nodes[net.ipaths[0].dst[0]].isize)
		dst = factory.empty(net.nodes[net.opaths[0].src[0]].isize)

		ctx.src = src
		ctx.dst = dst

		imem = ctx.state.newMemory(factory)
		ierr = ctx.state.newError(factory)
		trace_stack = [net.newTrace(factory) for i in range(self.maxlen)]

		try:
			batch = [next(data) for _ in range(self.size)]
		except StopIteration:
			pass

		for series in batch:
			ctx.setmem(imem)  # TODO: use set(other) instead

			for l, entry in enumerate(series):
				entry.getinput(src)

				# feedforward
				net.transmit(ctx)
				ctx.trace.copyto(trace_stack[l])

			ctx.seterr(ierr)

			for l, entry in reversed(list(enumerate(series))):
				entry.getouptut(dst)
				a = np.argmax(dst.np)
				ctx.nodes[8].target = a

				# backpropagate
				trace_stack[l].copyto(ctx.trace)
				net.backprop(ctx)

				loss += ctx.nodes[net.opaths[0].src[0]].loss

		ctx.grad.mul(1/self.size)
		return loss/self.size
