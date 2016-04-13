#!/usr/bin/python3


class BatchInfo:
	def __init__(self, size, callback=None):
		self.size = size
		self.callback = callback


class Teacher:
	def __init__(self, factory, batch, net, state=None, **opts):
		self.factory = factory
		self.batch = batch

		self.net = net
		net.prepare()

		if state is None:
			state = net.newState(factory)
		self.state = state

		self.ctx = net.newContext(factory)
		self.ctx.state = state
		self.ctx.trace = net.newTrace(factory)
		self.ctx.grad = state.newGradient(factory)
		self.ctx.rate = state.newRate(
			factory, opts.get('rate', 1e-1),
			adagrad=opts.get('adagrad', True)
		)
		self.clip = opts.get('clip', 5e0)

		self.ctx.src = factory.empty(net.isize)
		self.ctx.dst = factory.empty(net.osize)

		self.imem = state.newMemory(factory)
		self.ierr = state.newError(factory)

	def _batch(self, diter):
		batch = []
		try:
			for _ in range(self.batch.size):
				batch.append(next(diter))
		except StopIteration:
			if len(batch) == 0:
				raise StopIteration()

		ctx = self.ctx
		ctx.grad.clear()
		ctx.loss = 0.

		for series in batch:
			ctx.setmem(self.imem)
			for i, entry in enumerate(series):
				entry.getinput(ctx.src)
				self.net.transmit(ctx)
				self.traces[i].set(ctx.trace)

			ctx.seterr(self.ierr)
			for i, entry in reversed(list(enumerate(series))):
				entry.getouptut(ctx.dst)
				ctx.trace.set(self.traces[i])
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
				if self.batch.callback is not None:
					self.batch.callback(self.ctx)
			except StopIteration:
				break
