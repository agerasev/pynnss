#!/usr/bin/python3


class Teacher:
	def __init__(self, factory, data, net, state=None, **kwagrs):
		self.factory = factory

		self.data = data

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
			factory, kwagrs.get('rate', 1e-1),
			adagrad=kwagrs.get('adagrad', True)
		)
		self.clip = kwagrs.get('clip', 5e0)

		self.traces = None
		maxlen = kwagrs.get('maxlen', 1)
		if maxlen > 1:
			self.traces = [self.net.newTrace(self.factory) for _ in range(maxlen)]

		self.ctx.src = factory.empty(net.isize)
		self.ctx.dst = factory.empty(net.osize)

		self.imem = state.newMemory(factory)
		self.ierr = state.newError(factory)

		self.teachgen = self._TeachGen()

		self.bmon = kwagrs.get('bmon', None)
		self.emon = kwagrs.get('emon', None)

	def _batch(self, batch):
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

	def _EpochGen(self, epoch):
		for batch in epoch:
			self._batch(batch)
			try:
				if self.bmon is not None:
					self.bmon(self.ctx)
			except StopIteration:
				yield

	def _TeachGen(self):
		for epoch in self.data:
			epochgen = self._EpochGen(epoch)
			try:
				while True:
					next(epochgen)
					yield
			except StopIteration:
				try:
					if self.emon is not None:
						self.emon(self.ctx)
				except StopIteration:
					yield

	def teach(self):
		next(self.teachgen)
