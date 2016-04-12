#!/usr/bin/python3


class Teacher:
	def __init__(
		self, factory, bsize, net, state=None,
		rate=1e-1, adagrad=True, clip=5e0
	):
		self.factory = factory
		self.bsize = bsize

		self.net = net
		if state is None:
			state = net.newState(factory)
		self.state = state

		self.ctx = net.newContext(factory)
		self.ctx.state = state
		self.ctx.trace = net.newTrace(factory)
		self.ctx.grad = state.newGradient(factory)
		self.ctx.rate = state.newRate(factory, rate, adagrad=adagrad)
		self.clip = clip

		self.ctx.src = factory.empty(net.isize)
		self.ctx.dst = factory.empty(net.osize)

		self.imem = state.newMemory(factory)
		self.ierr = state.newError(factory)

	def _batch(self, diter):
		batch = []
		try:
			for _ in range(self.bsize):
				batch.append(next(diter))
		except StopIteration:
			if len(batch) == 0:
				raise StopIteration()

		ctx = self.ctx
		ctx.grad.clear()
		for series in batch:
			ctx.setmem(self.imem)
			for l, entry in enumerate(series):
				entry.getinput(ctx.src)
				self.net.transmit(ctx)
				ctx.trace.copyto(self.traces[l])

			ctx.seterr(self.ierr)
			for l, entry in reversed(list(enumerate(series))):
				entry.getouptut(ctx.dst)
				self.traces[l].copyto(ctx.trace)
				self.net.backprop(ctx)

		ctx.grad.mul(1/len(batch))
		ctx.grad.clip(self.clip)
		if hasattr(ctx.rate, 'update'):
			ctx.rate.update(ctx.grad)
		ctx.state.learn(ctx.grad, ctx.rate)

	def epoch(self, data, maxlen):
		self.traces = [self.net.newTrace(self.factory) for i in range(maxlen)]

		diter = iter(data)
		while True:
			try:
				self._batch(diter)
			except StopIteration:
				break
