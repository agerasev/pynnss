#!/usr/bin/python3

# Node is structural unit of NN

class Site:
	def __init__(self, size):
		self.size = size
		self.data = None
	def _repr(self):
		return 'size: ' + str(self.size) + ', data: ' + str(self.data)
	def __repr__(self):
		return '{' + self._repr() + '}'

class Node:
	# nin/nout - number of inputs/outputs
	def __init__(self, nin, nout):
		self.ins = [None]*nin
		self.outs = [None]*nout

	def _pre(self):
		for site in self.outs:
			assert site.data is None, 'out site is not empty'
	def _post(self):
		for site in self.ins:
			site.data = None
	def _step(self):
		raise NotImplementedError()
	def step(self):
		self._pre()
		self._step()
		self._post()
		
	def clear(self):
		raise NotImplementedError()

	def _repr(self):
		return 'ins: ' + str(self.ins) + ', outs: ' + str(self.outs)
	def __repr__(self):
		return '{' + self._repr() + '}'

class ElemNode(Node):
	# lsin/lsout - lists of input/output sizes
	def __init__(self, lsin, lsout):
		nin = len(lsin)
		nout = len(lsout)
		Node.__init__(self, nin, nout)
		for i in range(nin):
			self.ins[i] = Site(lsin[i])
		for i in range(nout):
			self.outs[i] = Site(lsout[i])

	def step(self):
		Node._pre(self)
		found = False
		for site in self.ins:
			if site.data is not None:
				found = True
				break
		if found:
			self._step()
			Node._post(self)

	def clear(self):
		pass

import numpy as np

class ConvNode(ElemNode):
	# sin/sout - size of inputs/outputs
	def __init__(self, sin, sout):
		ElemNode.__init__(self, [sin], [sout])
		self.randomize()

	def _step(self):
		self.outs[0].data = np.dot(self.weight, self.ins[0].data) + self.bias

	def randomize(self):
		sin = self.ins[0].size
		sout = self.outs[0].size
		self.weight = 2*np.random.rand(sin, sout) - 1
		self.bias = 2*np.random.rand(sout) - 1

	def _repr(self):
		rs = ''
		rs += ElemNode._repr(self) + ',\n'
		rs += 'weight:\n' + str(self.weight) + ',\nbias: ' + str(self.bias)
		return rs
	def __repr__(self):
		return '{\n' + self._repr() + '\n}'
