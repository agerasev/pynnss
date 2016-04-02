#!/usr/bin/python3

from dutil import dcopyto

if __name__ != '__main__':
	from pynn import prof
else:
	import prof


# Node is structural unit of network
class Node:
	# describes node sites
	class Site:
		def __init__(self, size, dtype=float):
			self.size = size
			self.dtype = dtype

		def _count(site):
			if isinstance(site, Node.Site):
				return 1
			else:
				for s in site:
					if not isinstance(s, Node.Site):
						raise Exception('arg is not site or sequence of sites')
				return len(site)

	# stores node state like biases or matrix weights
	class _State:
		def __init__(self):
			pass

		def copyto(self, out):
			raise NotImplementedError()

		# stores learn results
		class _Gradient:
			def __init__(self):
				pass

			def mul(self, factor):
				raise NotImplementedError()

			def clip(self, value):
				raise NotImplementedError()

		def newGradient(self):
			return None

		def learn(self, grad, rate):
			raise NotImplementedError()

	def newState(self):
		return None

	# stores node memory e.g. inputs, outputs and data in loopbacks
	class _Memory:
		def __init__(self, din=None, dout=None):
			self.din = din
			self.dout = dout

		def copyto(self, out):
			dcopyto(out.din, self.din)
			dcopyto(out.dout, self.dout)

	def newMemory(self):
		return None

	# stores node errors
	class _Error:
		def __init__(self, ein=None, eout=None):
			self.ein = ein
			self.eout = eout

		def copyto(self, out):
			dcopyto(out.ein, self.ein)
			dcopyto(out.eout, self.eout)

	def newError(self):
		return None

	# context for evaluation
	class Context:
		def __init__(self, din, dout, **kwargs):
			self.din = din
			self.dout = dout
			self.state = kwargs.get('state')
			self.mem = kwargs.get('mem')

	class ContextLearn(Context):
		def __init__(self, din, dout, ein, eout, **kwargs):
			self.Context.__init__(self, din, dout, **kwargs)
			self.ein = ein
			self.eout = eout
			self.grad = kwargs.get('grad')
			self.err = kwargs.get('err')
			self.rate = kwargs.get('rate')

	def __init__(self, isite, osite, **kwargs):
		# check and set in/out info
		self.isite = isite
		self.nin = Node.Site._count(isite)
		self.osite = osite
		self.nout = Node.Site._count(osite)

		if kwargs.get('prof', False):
			self.fstat = prof.Time()
			self.bstat = prof.Time()
		else:
			self.fstat = prof.Empty()
			self.bstat = prof.Empty()

	def transmit(self, ctx):
		with self.fstat:
			self._transmit(ctx)
		mem = ctx.mem
		if mem is not None:
			dcopyto(mem.din, ctx.din)
			dcopyto(mem.dout, ctx.dout)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def backprop(self, ctx):
		with self.bstat:
			self._backprop(ctx)
		err = ctx.err
		if err is not None:
			dcopyto(err.ein, ctx.ein)
			dcopyto(err.eout, ctx.eout)

	def _backprop(self, ctx):
		raise NotImplementedError()
