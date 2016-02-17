#!/usr/bin/python3

# Node is structural unit of NN

class NodeError(Exception):
	def __init__(self, value):
		Exception.__init__(self)
		self.value = value
	def __str__(self):
		return repr(self.value)

class Node:
	def __init__(self, nin, nout):
		self.inputs = [None]*nin
		self.outputs = [None]*nout

	def _step(self):
		raise NotImplementedError()

	def step(self):
		# check at least one input is not None
		found = False
		for signal in self.inputs:
			if signal is not None:
				found = True
				break
		if not found:
			return False

		# check all outputs is None
		found = False
		for signal in self.outputs:
			if signal is not None:
				found = True
				break
		if found:
			raise NodeError('outputs are not None : step will cause loss of data')

		# transmit signal
		self._step()
		return True

	def _repr(self):
		return 'inputs: ' + str(self.inputs) + ',\noutputs: ' + str(self.outputs)
	def __repr__(self):
		return '{\n' + self._repr() + '\n}'

import numpy as np

class ConvNode(Node):
	def __init__(self, nin, nout):
		Node.__init__(self, nin, nout)
		self.weight = np.zeros((nin, nout))
		self.bias = np.zeros((nout))

	def _repr(self):
		return Node._repr(self) + ',\nweight:\n' + str(self.weight) + ',\nbias: ' + str(self.bias)
	def __repr__(self):
		return '{\n' + self._repr() + '\n}'
